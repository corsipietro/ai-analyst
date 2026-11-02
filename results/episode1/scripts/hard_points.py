"""Keyword count of the hidden problems ("hard points") found in the final answers, per setup.

A raw, automatic count: for each (domain, question) the check passes when ALL its regular expressions match the
final answer (reasoning removed). Spot-checked by hand; a match does not prove the reasoning is right.
Labels are in Italian, as in the original count. v2 criteria (1/11/2026): 4 criteria corrected after a manual
spot check (false negatives on "21 runs" and ST01, false positives on Fiumicino and manual stations).

Run from results/episode1/runs:
    python3 ../scripts/hard_points.py react rewoo repair react_26b rewoo_26b repair_26b
"""
import collections
import glob
import json
import re
import sys

def final(f):
    """The last model message of a transcript, without the [REASONING] blocks."""
    t=open(f).read(); l=t.rsplit('================================== Ai Message ==================================',1)[-1]
    return re.sub(r'\[REASONING\].*?\[/REASONING\]','',l,flags=re.S)
checks={ # (domain, question prefix): [(label, [regexes that must ALL match])]
 ('toy_shell','3'):[('S*=400 e L*~80',[r'\b400\b',r'\b80\b'])],
 ('toy_shell','1'):[('21 run / 7x3',[r'\b21\b|7 .{0,20}3 (rep|run|seed)|\b3 (different |random |independent )?(seeds|repetitions|runs)|(three|3) (independent )?(runs|repetitions|seeds)']),('non monotono 60>70',[r'non.?monoton|dip|lower than .{0,30}60|975'])],
 ('toy_shell','2'):[('non monotono = rumore',[r'non.?monoton|dip|975',r'noise|not (statistically )?significant|uncertain'])],
 ('gen_rentals','3'):[('quota 56,5 con orfani',[r'56\.5'])],
 ('gen_rentals','1'):[('ST01 duplicato',[r'ST01|duplicate (entry|station|row)']),('orfani',[r'orphan|not exist|no match|not in'])],
 ('gen_rentals','2'):[('orfani citati',[r'orphan|not exist|no match|not linked'])],
 ('gen_app','1'):[('NULL = nessun acquisto',[r'no purchase|did not (make|result)|not result in a purchase|without (a )?purchase']),('utenti su piu dispositivi',[r'192|several devices|multiple devices|more than one device|overlap'])],
 ('gen_app','3'):[('per sessione 0.31/0.21/0.41',[r'0\.3[01]',r'0\.21',r'0\.4[01]'])],
 ('gen_energy','3'):[('blocco ID04 9-18 ago',[r'ID04|Sangro',r'(9|10).{0,15}18'])],
 ('gen_energy','1'):[('blocco ID04',[r'ID04|Sangro']),('capacita/potenza',[r'capacity|potenza|installed power|per kW'])],
 ('gen_energy','2'):[('tutti e 4 i numeri',[r'\b62\b',r'\b59\b',r'3[,.]?29[78]',r'55[,.]?89'])],
 ('gen_weather','1'):[('stazioni manuali',[r'manual|(Cassino|Tivoli|Terminillo).{0,150}(wind|missing|100)']),('guasto Fiumicino',[r'Fiumicino.{0,120}(gap|missing|fail|fault|outage)|(gap|missing|fail|fault|outage).{0,120}Fiumicino']),('quota/altitudine',[r'altitud|elevation|quota'])],
 ('gen_weather','3'):[('manuali 366',[r'manual'])],
 ('gen_weather','2'):[('7 giorni',[r'\b7\b'])],
}

def main():
    res=collections.defaultdict(collections.Counter)
    ARCHS=sys.argv[1:] or ['react','rewoo','repair','react_26b','rewoo_26b','repair_26b']
    for arch in ARCHS:
      for d in sorted(glob.glob(f'*_g5_{arch}')):
        dom=json.loads(open(d+'/runs.jsonl').readline())['domain']
        for f in sorted(glob.glob(d+'/*_r*.txt')):
            q=f.split('/')[-1][0]; t=final(f)
            for lab,pats in checks.get((dom,q),[]):
                res[(dom,lab)][arch]+= all(re.search(p,t,re.I) for p in pats); res[(dom,lab)]['n_'+arch]+=1
    for (dom,lab),c in res.items():
        print(f"{dom:12} {lab:28} " + "  ".join(f"{a} {c[a]}/{c['n_'+a]}" for a in ARCHS))
    tot={a:(sum(c[a] for c in res.values()),sum(c['n_'+a] for c in res.values())) for a in ARCHS}
    print('TOTAL', '  '.join(f"{a} {x}/{n}" for a,(x,n) in tot.items()))


if __name__ == "__main__":
    main()
