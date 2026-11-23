import json, time, urllib.request, subprocess, sys
U="http://localhost:8040"
def post(model):
    b=json.dumps({"prompt":"Name a city worth visiting and say why, in one sentence.","n":20,"t_prime":1.0,"max_new_tokens":64,"seed":1,"model":model}).encode()
    t=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(U+"/sample",b,{"Content-Type":"application/json"}),timeout=400)
        d=json.load(r); return r.status,time.time()-t,d
    except Exception as e:
        return str(e),time.time()-t,None
st=json.load(urllib.request.urlopen(U+"/status"))
keys=[m["key"] for m in st["models"]]
def mem():
    return subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",shell=True,capture_output=True,text=True).stdout.strip().replace("\n"," | ")
for k in keys:
    s1,t1,d1=post(k); s2,t2,d2=post(k)
    if d2:
        S=d2["samples"]; dif=max(abs(s["logp"]-s["logq"]) for s in S)
        print(k,s1,"cold %.1f warm %.1f"%(t1,t2),"modelKey",d2["modelKey"],"ended %d/%d"%(sum(s["ended"] for s in S),len(S)),"distinct",len({s["text"] for s in S}),"maxdiff %.2e"%dif,"| "+mem(),flush=True)
    else: print(k,s1,s2,flush=True)
