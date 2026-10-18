"""Standalone scientific figures using a private, isolated Matplotlib tool."""
import argparse,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'ROOT_private_plot_tool_v1'))
os.environ['MPLCONFIGDIR']=str(D/'ROOT_private_plot_tool_v1/mpl-config')
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    data=json.loads(args.data.read_text());assert data['whole463_AR_qualified']is True
    assert not args.output.exists();args.output.mkdir()
    names={'tight':'Coding / tight SLO','normal':'Chat / normal SLO','loose':'Summarization / loose SLO'}
    for cls,row in data['curves'].items():
        fig,ax=plt.subplots(figsize=(6.5,4.1),layout='constrained')
        ax.plot(row['n'],row['mean_delivered'],marker='o',color='#176b98',label='Same-prefix mean')
        ax.fill_between(row['n'],*row['request_cluster_bootstrap95'],alpha=.15,color='#176b98',label='Request-cluster bootstrap 95%')
        ax.axvline(4,color='#666666',linestyle='--',linewidth=1)
        ax.set(xlabel='Selected speculative candidates per request, n',ylabel='Delivered tokens per verification',
            title=names[cls],xticks=list(range(1,9)),ylim=(1,5.1))
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        ax.text(.98,.03,f"B8 cohort: {row['B8_observed_requests']} requests, {row['B8_untruncated_verifications']} verifications\nB1-7 are exact prefix counterfactuals; timings are not compared",transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
        for suffix in('png','svg'):fig.savefig(args.output/f'{cls}-same-prefix-gain.{suffix}',dpi=180)
        plt.close(fig)
    (args.output/'SOURCE.json').write_text(json.dumps(dict(data=str(args.data),snapshot_sha256=data['source_sha256'],matplotlib=matplotlib.__version__,model_or_environment_changes=False,source_only_private_tool=True),indent=2)+'\n')
    print(args.output)
if __name__=='__main__':main()
