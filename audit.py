"""Generic statistical diagnostics; input observations must be ordered in time."""
import numpy as np

def moving_block_mean_ci(values, block_length=10, iterations=5000, seed=42, confidence=.95):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError("finite one-dimensional sample required")
    if not 1<=block_length<=len(x) or iterations<1 or not 0<confidence<1: raise ValueError("invalid bootstrap configuration")
    rng=np.random.default_rng(seed); n=len(x); means=np.empty(iterations)
    for i in range(iterations):
        starts=rng.integers(0,n-block_length+1,size=int(np.ceil(n/block_length)))
        idx=np.concatenate([np.arange(s,s+block_length) for s in starts])[:n]; means[i]=x[idx].mean()
    tail=(1-confidence)/2
    return {"mean":float(x.mean()),"lower":float(np.quantile(means,tail)),"upper":float(np.quantile(means,1-tail)),"n":n,"block_length":block_length,"iterations":iterations}

def newey_west_mean_se(values, lags):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or not 0<=lags<len(x): raise ValueError("invalid observations or lags")
    z=x-x.mean();n=len(x);variance=np.dot(z,z)/n
    for k in range(1,lags+1): variance+=2*(1-k/(lags+1))*np.dot(z[k:],z[:-k])/n
    return float(np.sqrt(max(variance,0)/n))

def purged_split(feature_time,label_end,validation_start):
    feature_time=np.asarray(feature_time);label_end=np.asarray(label_end)
    if len(feature_time)!=len(label_end) or np.any(label_end<feature_time): raise ValueError("invalid label window")
    train=np.flatnonzero((feature_time<validation_start)&(label_end<validation_start));valid=np.flatnonzero(feature_time>=validation_start)
    return train,valid


def rank_ic(signal,forward_return):
    import pandas as pd
    x=np.asarray(signal,dtype=float);y=np.asarray(forward_return,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or len(x)!=len(y):raise ValueError("matching one-dimensional arrays required")
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3:return float("nan")
    return float(pd.Series(x[ok]).rank(method="average").corr(pd.Series(y[ok]).rank(method="average")))
