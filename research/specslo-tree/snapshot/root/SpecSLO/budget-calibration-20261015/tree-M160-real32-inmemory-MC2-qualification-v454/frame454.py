"""HOST slot binding for eight genuinely separate, prefilled reference owners."""


def extra_slots(positions,block_tables,prefix_lengths):
    if len(positions)!=120 or len(prefix_lengths)!=24:raise ValueError('original24 real q5 groups')
    owners=list(range(24,32));slots=[]
    for i,owner in enumerate(owners):
        for j in range(5):
            pos=positions[5*i+j]
            if type(pos)is not int or pos!=prefix_lengths[i]+j or not 0<=pos<2560:
                raise ValueError('true same-prefix query positions, no invented roots')
            page=block_tables[owner][pos//128]
            if type(page)is not int or not 0<=page<544:raise ValueError('all real32 writes pre-reserved')
            slots.append(page*128+pos%128)
    if len(slots)!=40 or len(set(slots))!=40:raise ValueError('forty distinct additional real writes')
    return owners,slots
