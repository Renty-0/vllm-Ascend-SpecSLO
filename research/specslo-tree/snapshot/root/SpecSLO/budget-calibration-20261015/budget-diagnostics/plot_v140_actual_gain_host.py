#!/usr/bin/env python3
"""Plot retained actual_gain_state only after the requested phase is complete.

Stdlib for validation/CSV; existing Matplotlib Agg, or system Pillow + SVG, for plots. No candidate,
Torch or device imports/queries. Writes only a new child directory beside this
script. Does not run a workload or grant numerical/Goodput qualification.
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

D=Path(__file__).resolve().parent
CLASSES=('tight','normal','loose')
BS=(1,2,3,4,5)
PHASES=('warmup65','zero65-1','zero65-2','full463')
MAX_FILE=64*1024**2
MAX_EXPANDED=256*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def file_record(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def read_json(path):
    path=Path(path);require(path.is_file() and 0<path.stat().st_size<=MAX_FILE,'Missing/oversized input: '+str(path))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as stream:raw=stream.read(MAX_EXPANDED+1)
    require(len(raw)<=MAX_EXPANDED,'Expanded JSON exceeds bound: '+str(path))
    def reject(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(raw,parse_constant=reject)


def safe_artifact(artifact,run_root):
    require(isinstance(artifact,dict),'Phase artifact reference is missing')
    path=Path(artifact['path']).resolve()
    require(path.is_relative_to(run_root) and path.is_file(),'Phase artifact escaped its actual run')
    require(path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256'],'Phase artifact length/SHA mismatch')
    value=read_json(path)
    require(isinstance(value,dict),'Phase artifact is not a complete JSON object')
    return path,value


def normalize_gain(gain,generation):
    require(isinstance(gain,dict) and gain.get('actual_engine_generation')==generation,'Actual gain snapshot generation mismatch')
    require(gain.get('prior_weight')==12 and gain.get('unobserved_B_filled_with_zero') is False
        and gain.get('counter_delta_or_cross_generation_subtraction') is False,'Unexpected actual policy gain snapshot contract')
    observed=gain.get('observed');base=gain.get('base')
    require(isinstance(observed,list) and isinstance(base,list),'Actual observed/base arrays are missing')
    observations={};priors={}
    for row in observed:
        require(isinstance(row,dict) and row.get('slo_class') in CLASSES and type(row.get('B')) is int
            and row['B'] in BS and type(row.get('rounds')) is int and row['rounds']>0
            and type(row.get('delivered')) is int and row['delivered']>=0,'Invalid actual observed class/B sample')
        key=row['slo_class'],row['B'];require(key not in observations,'Duplicate observed class/B')
        observations[key]=(row['rounds'],row['delivered'])
    for row in base:
        require(isinstance(row,dict) and row.get('slo_class') in CLASSES and isinstance(row.get('values'),list),'Invalid base class')
        for item in row['values']:
            require(isinstance(item,dict) and type(item.get('B')) is int and 1<=item['B']<=8
                and type(item.get('g')) in (int,float) and math.isfinite(item['g']) and item['g']>=0,'Invalid actual base prior')
            key=row['slo_class'],item['B'];require(key not in priors,'Duplicate prior class/B');priors[key]=item['g']
    require(all((cls,b) in priors for cls in CLASSES for b in BS),'Actual prior class/B domain incomplete')
    return observations,priors


def inputs(directory,tag,phase,rank):
    p=Path(directory).resolve();run_root=(p/'local-output'/tag).resolve()
    receipt_path=p/f'local-receipt-{tag}.json';receipt=read_json(receipt_path)
    require(isinstance(receipt,dict) and receipt.get('action')=='split192-candidate-full463','Wrong candidate action/receipt')
    if phase=='full463':
        require(receipt.get('status')=='complete' and receipt.get('errors')==[]
            and receipt.get('experimental_sources_unchanged') is True and receipt.get('external_sources_unchanged') is True,
            'full463 plots require the original terminal COMPLETE/error-free/unchanged-source receipt')
    else:require(receipt.get('status') in ('running','complete','failed'),'Unknown actual run status')
    snapshots=[];records=[];generations=[];scope=463 if phase=='full463' else 65
    for current_rank in range(4):
        final=run_root/f'rank{current_rank}-local-diagnostic.json'
        checkpoint=run_root/f'rank{current_rank}-candidate-checkpoint.json.gz'
        report_path=final if final.is_file() else checkpoint
        require(phase!='full463' or report_path==final,'full463 requires final original worker reports, not partial checkpoints')
        report=read_json(report_path)
        require(isinstance(report,dict) and report.get('rank')==current_rank,'Worker report/checkpoint rank mismatch')
        if phase=='full463':require(report.get('full463_completed') is report.get('sealed') is report.get('cache_released') is True,'Final full worker scope is incomplete')
        traces=report.get('traces');require(isinstance(traces,list),'Original traces are absent')
        selected=[row for row in traces if isinstance(row,dict) and row.get('role')==phase]
        require(len(selected)==1,'Exactly one retained requested phase per rank is required')
        trace=selected[0];generation=trace.get('source_generation')
        require(type(generation) is int and generation>0 and trace.get('scope_count')==scope,'Phase scope/generation is invalid')
        artifact_path,raw=safe_artifact(trace.get('artifact'),run_root)
        require(raw.get('rank')==current_rank and raw.get('role')==phase and raw.get('scope_count')==scope
            and raw.get('source_generation')==generation and raw.get('cache_released') is True,'Actual artifact phase/generation is incomplete')
        if phase=='full463':require(raw.get('sealed_before') is raw.get('sealed_after') is True,'Full artifact was not sealed')
        if current_rank==1:
            results=raw.get('generate_results')
            require(isinstance(results,list) and len(results)==scope
                and all(isinstance(row.get('completion_token_ids'),list) and len(row['completion_token_ids'])==256 for row in results),
                'Target-leader output scope is incomplete')
        state=trace.get('actual_gain_state')
        require(isinstance(state,dict) and set(state)=={'before','after'},'Original completed phase gain snapshot pair is missing')
        obs,prior=normalize_gain(state['after'],generation)
        snapshots.append((obs,prior));generations.append(generation)
        records.append(dict(rank=current_rank,phase_report=file_record(report_path),phase_artifact=file_record(artifact_path)))
    require(len(set(generations))==1,'Four ranks disagree on actual phase generation')
    require(all(value==snapshots[rank] for value in snapshots),'Four cooperating ranks disagree on observed gain/prior state; do not pick or sum a rank')
    return dict(directory=str(p),tag=tag,phase=phase,scope_count=scope,source_generation=generations[0],rank_used=rank,
        receipt=file_record(receipt_path),source_run_status=receipt['status'],phase_inputs=records,
        cross_rank_gain_state_equal=True,four_ranks_are_independent_trials=False),snapshots[rank]


def table_rows(provenance,snapshot):
    obs,prior=snapshot;rows=[]
    for cls in CLASSES:
        for b in BS:
            actual=obs.get((cls,b));n,total=actual if actual else (None,None)
            rows.append(dict(phase=provenance['phase'],scope_count=provenance['scope_count'],
                source_generation=provenance['source_generation'],rank_used=provenance['rank_used'],slo_class=cls,B=b,
                observed_rounds=n,observed_delivered=total,observed_g=total/n if n is not None else None,
                policy_prior_g=prior[cls,b],policy_prior_weight=12,
                policy_intrinsic_posterior_g=(12*prior[cls,b]+total)/(12+n) if n is not None else None,
                actual_request_local_clamped_gain_reconstructed=False))
    return rows


def matplotlib_plots(rows,provenance,output,include_policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    artifacts=[]
    for cls in CLASSES:
        values=[r for r in rows if r['slo_class']==cls]
        fig,(ax,count_ax)=plt.subplots(2,1,figsize=(6.4,5.4),sharex=True,
            gridspec_kw={'height_ratios':[3,1.25]},constrained_layout=True)
        x=[r['B'] for r in values];y=[r['observed_g'] if r['observed_g'] is not None else math.nan for r in values]
        ax.plot(x,y,'o-',color='#1764ab',linewidth=1.6,label='Observed delivered / round')
        if include_policy:
            ax.plot(x,[r['policy_prior_g'] for r in values],'--',color='#777777',label='Policy calibrated prior (not observed)')
            ax.plot(x,[r['policy_intrinsic_posterior_g'] if r['policy_intrinsic_posterior_g'] is not None else math.nan for r in values],
                's-.',color='#c16e00',markersize=4,label='(12 × prior + delivered) / (12 + rounds)')
        for row in values:
            if row['observed_g'] is not None:
                ax.annotate(f"n={row['observed_rounds']:,}",(row['B'],row['observed_g']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        if not any(r['observed_rounds'] is not None for r in values):
            ax.text(.5,.5,'No observed samples in this class',transform=ax.transAxes,ha='center')
        ax.set_ylabel('Delivered tokens / verification round')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25);ax.legend(loc='best',fontsize=8)
        ax.set_title(f"{cls} | {provenance['phase']} | generation {provenance['source_generation']}")
        observed=[r for r in values if r['observed_rounds'] is not None]
        count_ax.bar([r['B'] for r in observed],[r['observed_rounds'] for r in observed],color='#9fc4df',width=.55)
        count_ax.set_ylabel('Observed\nrounds');count_ax.set_xlabel('Retained candidate budget B')
        count_ax.set_xticks(BS);count_ax.set_xlim(.5,5.5);count_ax.grid(axis='y',alpha=.2)
        count_ax.set_ylim(0,max([r['observed_rounds'] for r in observed],default=1)*1.25)
        for row in values:
            if row['observed_rounds'] is None:count_ax.text(row['B'],.08,'unobserved',transform=count_ax.get_xaxis_transform(),ha='center',rotation=90,fontsize=7,color='#777777')
        scope='FULL 463' if provenance['phase']=='full463' else '65-request phase only; NOT full463'
        fig.supxlabel(scope+' · missing B left blank · no CI · selected/correlated observations',fontsize=8)
        for extension in ('png','svg'):
            path=output/f'{cls}-observed-g.{extension}'
            fig.savefig(path,dpi=220 if extension=='png' else None)
            artifacts.append(file_record(path))
        plt.close(fig)
    return artifacts,matplotlib.__version__



def pillow_svg_plots(rows,provenance,output,include_policy):
    """Same vector coordinates for SVG and existing Pillow raster, no install."""
    import PIL
    from PIL import Image,ImageDraw,ImageFont
    from xml.sax.saxutils import escape
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font_family='DejaVu Sans' if font_path.is_file() else 'sans-serif'
    width,height=1280,1040
    artifacts=[]
    for cls in CLASSES:
        image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
        fonts={}
        def text(x,y,value,size=23,color='#222222',anchor='left'):
            value=str(value)
            if size not in fonts:fonts[size]=ImageFont.truetype(str(font_path),size) if font_path.is_file() else ImageFont.load_default(size=size)
            font=fonts[size]
            box=draw.textbbox((0,0),value,font=font);span=box[2]-box[0]
            left=x-span/2 if anchor=='middle' else x-span if anchor=='end' else x
            draw.text((left,y),value,fill=color,font=font,anchor='lt')
            svg_anchor='middle' if anchor=='middle' else 'end' if anchor=='end' else 'start'
            svg.append(f'<text x="{x:.3f}" y="{y+size:.3f}" font-family="{font_family}" font-size="{size}" fill="{color}" text-anchor="{svg_anchor}">{escape(value)}</text>')
        def line(x1,y1,x2,y2,color='#222222',thick=2,dashed=False):
            if dashed:
                length=math.hypot(x2-x1,y2-y1)
                if length:
                    for start in range(0,int(length)+1,18):
                        a=start/length;b=min(start+10,length)/length
                        draw.line((x1+a*(x2-x1),y1+a*(y2-y1),x1+b*(x2-x1),y1+b*(y2-y1)),fill=color,width=thick)
            else:draw.line((x1,y1,x2,y2),fill=color,width=thick)
            dash=' stroke-dasharray="10 8"' if dashed else ''
            svg.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{thick}"{dash}/>')
        def rectangle(x1,y1,x2,y2,color):
            draw.rectangle((x1,y1,x2,y2),fill=color)
            svg.append(f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{x2-x1:.3f}" height="{y2-y1:.3f}" fill="{color}"/>')
        def point(x,y,color,square=False):
            if square:rectangle(x-6,y-6,x+6,y+6,color)
            else:
                draw.ellipse((x-7,y-7,x+7,y+7),fill=color)
                svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="7" fill="{color}"/>')
        values=[r for r in rows if r['slo_class']==cls]
        observed=[r for r in values if r['observed_rounds'] is not None]
        xleft,xright=140,1210;top,bottom=165,625;ctop,cbottom=755,915
        xp=lambda b:xleft+(b-.5)/5*(xright-xleft)
        candidates=[r['observed_g'] for r in observed]
        if include_policy:candidates += [r['policy_prior_g'] for r in values]+[r['policy_intrinsic_posterior_g'] for r in observed]
        ymax=max(1,math.ceil(max(candidates,default=1)+.4));yp=lambda g:bottom-(bottom-top)*g/ymax
        countmax=max([r['observed_rounds'] for r in observed],default=1)
        countstep=max(1,math.ceil(countmax/4));counttop=4*countstep;cy=lambda n:cbottom-(cbottom-ctop)*n/counttop
        text(width/2,23,f"{cls} | {provenance['phase']} | generation {provenance['source_generation']}",31,anchor='middle')
        line(145,93,200,93,'#1764ab',4);point(173,93,'#1764ab');text(215,77,'Observed delivered / round',21)
        if include_policy:
            line(630,93,685,93,'#777777',3,True);text(700,77,'Calibrated prior (not observed)',21)
            line(145,132,200,132,'#c16e00',3,True);text(215,116,'Intrinsic policy posterior: (12 prior + delivered) / (12 + rounds)',20)
        text(xleft,top-32,'Delivered tokens / verification round',23)
        for tick in range(ymax+1):
            y=yp(tick);line(xleft,y,xright,y,'#dddddd',1);text(xleft-16,y-13,tick,21,anchor='end')
        line(xleft,top,xleft,bottom);line(xleft,bottom,xright,bottom)
        def series(field,color,square=False,dashed=False):
            previous=None
            for row in values:
                value=row[field]
                if value is None:previous=None;continue
                x,y=xp(row['B']),yp(value)
                if previous:line(*previous,x,y,color,3,dashed)
                point(x,y,color,square=square);previous=x,y
        if include_policy:
            series('policy_prior_g','#777777',dashed=True)
            series('policy_intrinsic_posterior_g','#c16e00',square=True,dashed=True)
        series('observed_g','#1764ab')
        for row in observed:text(xp(row['B']),yp(row['observed_g'])-37,f"n={row['observed_rounds']:,}",20,anchor='middle')
        if not observed:text((xleft+xright)/2,(top+bottom)/2,'No observed samples in this class',25,anchor='middle')
        text(xleft,ctop-36,'Observed verification rounds (n)',23)
        for tick in range(0,counttop+1,countstep):
            y=cy(tick);line(xleft,y,xright,y,'#e7e7e7',1);text(xleft-16,y-13,f'{tick:,}',20,anchor='end')
        for row in observed:rectangle(xp(row['B'])-48,cy(row['observed_rounds']),xp(row['B'])+48,cbottom,'#9fc4df')
        line(xleft,ctop,xleft,cbottom);line(xleft,cbottom,xright,cbottom)
        for row in values:
            x=xp(row['B']);line(x,cbottom,x,cbottom+8);text(x,cbottom+14,row['B'],23,anchor='middle')
            if row['observed_rounds'] is None:text(x,ctop+65,'unobserved',18,'#777777',anchor='middle')
        text((xleft+xright)/2,cbottom+53,'Retained candidate budget B',25,anchor='middle')
        scope='FULL 463' if provenance['phase']=='full463' else '65-request phase only; NOT full463'
        text(width/2,992,scope+'; missing B left blank; no CI; selected/correlated observations',18,anchor='middle')
        svg.append('</svg>')
        svg_path=output/f'{cls}-observed-g.svg'
        with svg_path.open('x') as stream:stream.write('\n'.join(svg)+'\n')
        png_path=output/f'{cls}-observed-g.png'
        with png_path.open('xb') as stream:image.save(stream,format='PNG',dpi=(220,220))
        artifacts += [file_record(png_path),file_record(svg_path)]
    return artifacts,'Pillow '+PIL.__version__+' + stdlib SVG (same coordinates)'


def plots(rows,provenance,output,include_policy):
    from importlib.util import find_spec
    if find_spec('matplotlib') is not None:return matplotlib_plots(rows,provenance,output,include_policy)
    require(find_spec('PIL') is not None,'No existing plotting backend. Use the system Python3.12 interpreter with existing Pillow; do not install packages.')
    return pillow_svg_plots(rows,provenance,output,include_policy)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--phase',choices=PHASES,default='full463')
    parser.add_argument('--rank',type=int,choices=range(4),default=0)
    parser.add_argument('--include-policy-estimates',action='store_true',help='Separately labelled prior and intrinsic posterior; never called EMA or actual request-clamped gain')
    parser.add_argument('--output-name',help='New simple directory name under budget-diagnostics; existing outputs are never overwritten')
    args=parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None,'Invalid private tag')
    name=args.output_name or ('V140-actual-g-'+args.tag+'-'+args.phase)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name) is not None,'Output must be a simple new directory name')
    output=D/name;require(not output.exists(),'Refusing to overwrite existing output directory')
    from importlib.util import find_spec
    require(find_spec('matplotlib') is not None or find_spec('PIL') is not None,
        'Use existing system Python3.12 with Pillow; no backend is installed in retained3.11')
    provenance,snapshot=inputs(args.directory,args.tag,args.phase,args.rank)
    rows=table_rows(provenance,snapshot)
    # Validate everything before any output is created; a plotting failure leaves
    # that attempt visible rather than overwriting it on a later rerun.
    output.mkdir(exist_ok=False)
    csv_path=output/'actual-observed-g.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifacts,version=plots(rows,provenance,output,args.include_policy_estimates)
    manifest=dict(schema='V140.actual-observed-g-plots.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_argv=__import__('sys').orig_argv,script=file_record(__file__),plotting_backend=version,provenance=provenance,
        outputs=[file_record(csv_path),*artifacts],policy_estimates_drawn=args.include_policy_estimates,
        definitions=dict(observed_g='actual observed delivered / actual observed rounds; no prior in this value',
            prior='actual saved calibrated policy.base[class][B], not current-generation measurement',
            intrinsic_posterior='(12*prior + observed delivered)/(12 + observed rounds), shown only at observed B',
            actual_gain_function='min(request remaining tokens, max(1.0, intrinsic posterior)); remaining-token state is not reconstructed',
            EMA='not plotted or fabricated; observed delivered/round is not an EMA',
            missing='unobserved rounds/delivered/g/posterior remain blank; observed zeros are not fabricated',
            CI='none; correlated adaptive observations do not supply an independent CI here'),
        whole_workload_AR_or_numerical_qualification_inferred=False,formal_Goodput=False,
        candidate_sources_modified=False,NPU_or_Torch_or_model_or_device_query_executed=False)
    with (output/'provenance.json').open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output_directory=str(output),phase=args.phase,generation=provenance['source_generation'],
        observed_sample_rows=sum(r['observed_rounds'] is not None for r in rows),CSV=str(csv_path),PNG_count=3,SVG_count=3,
        provenance_sha256=sha(output/'provenance.json'),formal_Goodput=False)))

if __name__=='__main__':main()
