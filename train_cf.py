"""Counterfactual mechanism evolution. Run from the repository root in WSL."""
import argparse
from datetime import datetime
from functools import partial
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from alphagen.data.expression import Feature
from alphagen_qlib.stock_data import FeatureType, StockData
from alpha_cf.alpha_pool import AlphaCFPool
from alpha_cf.expression import parse
from alpha_cf.trainer import AlphaCFTrainer, make_logger, save_json


# These are candidate formulas, not inherited weights or evaluation results.
INITIAL_EXPRS = [
    # Alpha158: supported kbar/price/rolling families from qlib/contrib/data/loader.py (epsilon adapted to this parser)
    "Div(Sub($close,$open),$open)",
    "Div(Sub($high,$low),$open)",
    "Div(Sub($close,$open),Add(Sub($high,$low),0.000000000001))",
    "Div(Sub($high,GetGreater($open,$close)),$open)",
    "Div(Sub($high,GetGreater($open,$close)),Add(Sub($high,$low),0.000000000001))",
    "Div(Sub(GetLess($open,$close),$low),$open)",
    "Div(Sub(GetLess($open,$close),$low),Add(Sub($high,$low),0.000000000001))",
    "Div(Sub(Sub(Mul(2.0,$close),$high),$low),$open)",
    "Div(Sub(Sub(Mul(2.0,$close),$high),$low),Add(Sub($high,$low),0.000000000001))",
    "Div($open,$close)",
    "Div($high,$close)",
    "Div($low,$close)",
    "Div($vwap,$close)",
    "Div(Ref($close,5),$close)",
    "Div(TsMean($close,5),$close)",
    "Div(TsStd($close,5),$close)",
    "Div(TsMax($high,5),$close)",
    "Div(TsMin($low,5),$close)",
    "Div(Sub($close,TsMin($low,5)),Add(Sub(TsMax($high,5),TsMin($low,5)),0.000000000001))",
    "TsCorr($close,Log(Add($volume,1.0)),5)",
    "TsCorr(Div($close,Ref($close,1)),Log(Add(Div($volume,Ref($volume,1)),1.0)),5)",
    "Div(TsMean($volume,5),Add($volume,0.000000000001))",
    "Div(TsStd($volume,5),Add($volume,0.000000000001))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),5),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),5),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),5),Add(TsSum(Abs(Sub($close,Ref($close,1))),5),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),5),Add(TsSum(Abs(Sub($close,Ref($close,1))),5),0.000000000001))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),5),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),5)),Add(TsSum(Abs(Sub($close,Ref($close,1))),5),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),5),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),5),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),5),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),5),0.000000000001))",
    "Div(Ref($close,10),$close)",
    "Div(TsMean($close,10),$close)",
    "Div(TsStd($close,10),$close)",
    "Div(TsMax($high,10),$close)",
    "Div(TsMin($low,10),$close)",
    "Div(Sub($close,TsMin($low,10)),Add(Sub(TsMax($high,10),TsMin($low,10)),0.000000000001))",
    "TsCorr($close,Log(Add($volume,1.0)),10)",
    "TsCorr(Div($close,Ref($close,1)),Log(Add(Div($volume,Ref($volume,1)),1.0)),10)",
    "Div(TsMean($volume,10),Add($volume,0.000000000001))",
    "Div(TsStd($volume,10),Add($volume,0.000000000001))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),10),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),10),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),10),Add(TsSum(Abs(Sub($close,Ref($close,1))),10),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),10),Add(TsSum(Abs(Sub($close,Ref($close,1))),10),0.000000000001))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),10),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),10)),Add(TsSum(Abs(Sub($close,Ref($close,1))),10),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),10),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),10),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),10),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),10),0.000000000001))",
    "Div(Ref($close,20),$close)",
    "Div(TsMean($close,20),$close)",
    "Div(TsStd($close,20),$close)",
    "Div(TsMax($high,20),$close)",
    "Div(TsMin($low,20),$close)",
    "Div(Sub($close,TsMin($low,20)),Add(Sub(TsMax($high,20),TsMin($low,20)),0.000000000001))",
    "TsCorr($close,Log(Add($volume,1.0)),20)",
    "TsCorr(Div($close,Ref($close,1)),Log(Add(Div($volume,Ref($volume,1)),1.0)),20)",
    "Div(TsMean($volume,20),Add($volume,0.000000000001))",
    "Div(TsStd($volume,20),Add($volume,0.000000000001))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),20),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),20),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),20),Add(TsSum(Abs(Sub($close,Ref($close,1))),20),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),20),Add(TsSum(Abs(Sub($close,Ref($close,1))),20),0.000000000001))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),20),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),20)),Add(TsSum(Abs(Sub($close,Ref($close,1))),20),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),20),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),20),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),20),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),20),0.000000000001))",
    "Div(Ref($close,30),$close)",
    "Div(TsMean($close,30),$close)",
    "Div(TsStd($close,30),$close)",
    "Div(TsMax($high,30),$close)",
    "Div(TsMin($low,30),$close)",
    "Div(Sub($close,TsMin($low,30)),Add(Sub(TsMax($high,30),TsMin($low,30)),0.000000000001))",
    "TsCorr($close,Log(Add($volume,1.0)),30)",
    "TsCorr(Div($close,Ref($close,1)),Log(Add(Div($volume,Ref($volume,1)),1.0)),30)",
    "Div(TsMean($volume,30),Add($volume,0.000000000001))",
    "Div(TsStd($volume,30),Add($volume,0.000000000001))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),30),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),30),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),30),Add(TsSum(Abs(Sub($close,Ref($close,1))),30),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),30),Add(TsSum(Abs(Sub($close,Ref($close,1))),30),0.000000000001))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),30),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),30)),Add(TsSum(Abs(Sub($close,Ref($close,1))),30),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),30),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),30),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),30),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),30),0.000000000001))",
    "Div(Ref($close,60),$close)",
    "Div(TsMean($close,60),$close)",
    "Div(TsStd($close,60),$close)",
    "Div(TsMax($high,60),$close)",
    "Div(TsMin($low,60),$close)",
    "Div(Sub($close,TsMin($low,60)),Add(Sub(TsMax($high,60),TsMin($low,60)),0.000000000001))",
    "TsCorr($close,Log(Add($volume,1.0)),60)",
    "TsCorr(Div($close,Ref($close,1)),Log(Add(Div($volume,Ref($volume,1)),1.0)),60)",
    "Div(TsMean($volume,60),Add($volume,0.000000000001))",
    "Div(TsStd($volume,60),Add($volume,0.000000000001))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),60),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),60),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),60),Add(TsSum(Abs(Sub($close,Ref($close,1))),60),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),60),Add(TsSum(Abs(Sub($close,Ref($close,1))),60),0.000000000001))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),60),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),60)),Add(TsSum(Abs(Sub($close,Ref($close,1))),60),0.000000000001))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),60),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),60),0.000000000001))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),60),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),60),0.000000000001))",
    # Handwritten reversal, volume confirmation and volatility-normalized momentum
    "Sub(0.0,Div(TsDelta($close,5),Ref($close,5)))",
    "Mul(Sub(0.0,Div(TsDelta($close,5),Ref($close,5))),Div($volume,Add(TsMean($volume,5),0.000000000001)))",
    "Div(TsDelta($close,5),Add(TsStd($close,5),0.000000000001))",
    "Sub(0.0,Div(TsDelta($close,10),Ref($close,10)))",
    "Mul(Sub(0.0,Div(TsDelta($close,10),Ref($close,10))),Div($volume,Add(TsMean($volume,10),0.000000000001)))",
    "Div(TsDelta($close,10),Add(TsStd($close,10),0.000000000001))",
    "Sub(0.0,Div(TsDelta($close,20),Ref($close,20)))",
    "Mul(Sub(0.0,Div(TsDelta($close,20),Ref($close,20))),Div($volume,Add(TsMean($volume,20),0.000000000001)))",
    "Div(TsDelta($close,20),Add(TsStd($close,20),0.000000000001))",
    "Sub(0.0,Div(TsDelta($close,40),Ref($close,40)))",
    "Mul(Sub(0.0,Div(TsDelta($close,40),Ref($close,40))),Div($volume,Add(TsMean($volume,40),0.000000000001)))",
    "Div(TsDelta($close,40),Add(TsStd($close,40),0.000000000001))",
    "Sub(0.0,Div(TsDelta($close,60),Ref($close,60)))",
    "Mul(Sub(0.0,Div(TsDelta($close,60),Ref($close,60))),Div($volume,Add(TsMean($volume,60),0.000000000001)))",
    "Div(TsDelta($close,60),Add(TsStd($close,60),0.000000000001))",
    # GFN / AlphaSAGE; data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/pool_9999.json
    "TsVar(Rank(Pow($low,0.5)),20)",
    "Div(Sub(0.5,$vwap),-10.0)",
    "Sub($vwap,$close)",
    "Sub(0.01,Ref(Abs(Rank(TsPctChange(Add(Sub($low,2.0),-2.0),40))),30))",
    "TsCov(Sub(TsDiv(Mul($close,-0.5),30),-1.0),TsKurt(Log(Pow(Greater(Add($open,-0.5),-1.0),$vwap)),30),40)",
    "Pow(0.01,TsDiv(Add(Inv(Greater(Rank(Mul(Inv($high),0.01)),$high)),Less(2.0,SLog1p($open))),40))",
    "Sub(30.0,Rank(TsPctChange(Inv(Less(TsDelta($vwap,40),$open)),40)))",
    "Sign(Sub(2.0,Pow($close,Rank(Greater(TsVar($volume,20),Pow(SLog1p(Div(1.0,Rank($high))),-0.01))))))",
    "Mul(-0.5,TsSkew(Less(TsMinDiff($high,20),$low),40))",
    "Sub(Sign(TsMaxDiff(Sub(Add(30.0,$vwap),$volume),40)),Less(-2.0,$open))",
    "Pow(0.5,$close)",
    "Pow($volume,SLog1p(TsDiv($volume,50)))",
    "Sub(Sub(-0.01,Rank(Mul(Rank(Pow(TsPctChange($low,40),-5.0)),-30.0))),5.0)",
    "TsCorr($vwap,Sub($close,5.0),20)",
    "Add(Add(TsPctChange(Div($volume,$vwap),20),-5.0),10.0)",
    "TsKurt(Mul(Greater(5.0,$high),Add(Mul(TsSkew($close,50),-2.0),0.5)),40)",
    "Div(-10.0,Add(TsMax($low,20),Pow(TsIr(Sub(5.0,Less($volume,Pow($high,5.0))),30),10.0)))",
    "TsSkew(TsWMA(Div(Pow(5.0,SLog1p($vwap)),0.01),50),40)",
    "Log(TsStd(Mul(0.01,$volume),50))",
    "TsIr($volume,20)",
    "TsPctChange(Pow(TsIr($high,50),-0.01),40)",
    "Log(Inv(Mul(Log(Pow($high,Sub(TsSkew(Mul($low,0.01),40),-0.01))),2.0)))",
    "Sub(-0.5,TsSkew(Sub($open,30.0),50))",
    "Mul(Sub(Greater(0.01,Add($vwap,-0.01)),Mul($volume,-10.0)),$high)",
    "Mul(Add(TsMinDiff($volume,20),-1.0),Mul(TsSum(Mul($open,0.01),20),Inv(Sub(2.0,$low))))",
    "Inv(Sub(-2.0,Mul(Add(TsRank(TsRank($high,10),10),Ref(TsDelta(Mul($volume,-0.01),20),40)),0.5)))",
    "Mul(-30.0,TsMin(TsSkew($close,10),50))",
    "TsDelta(SLog1p($volume),40)",
    "Div(Less(-5.0,$high),Add(TsCov(Pow($low,Greater(-1.0,TsMin(Rank(Mul($high,-30.0)),20))),$vwap,10),-2.0))",
    "Add(TsPctChange($close,20),Rank(TsIr(Inv(Greater(TsMinDiff($low,40),Div(Add(-10.0,$close),$close))),40)))",
    "Mul(SLog1p(Pow(TsSkew(Mul(Pow(0.5,$vwap),30.0),30),-0.5)),10.0)",
    "TsKurt(Mul(0.01,Div(Mul(Inv($high),-0.01),-5.0)),20)",
    "Inv(TsStd(Log(TsRank(Div(-5.0,$volume),40)),50))",
    "TsCov(Abs($volume),$low,30)",
    "Abs(Add(-2.0,TsKurt($open,50)))",
    "TsIr(SLog1p(TsCorr($vwap,$low,20)),30)",
    "Add(30.0,TsSkew(Sub(Pow($open,-1.0),Inv(Log(Add($close,1.0)))),20))",
    "Sub(-30.0,TsMad(Log(TsEMA($volume,20)),30))",
    "TsSkew(Sub(Div(-0.01,Sub(Pow($vwap,Mul(Less(-5.0,Sign($vwap)),$open)),$open)),0.01),10)",
    "Add(2.0,Sub(TsPctChange(Mul(TsVar($volume,20),$open),20),Less(1.0,$open)))",
    "Log(Div(0.5,TsSkew(Log(Mul($low,Rank($close))),50)))",
    "Rank(Pow(TsSkew(Sub(Log($close),1.0),40),Less(Rank($open),5.0)))",
    "Sub(-0.01,TsDelta(TsRank(Rank($low),10),50))",
    "TsSkew(Div(Ref($high,20),-0.01),40)",
    "TsMed(TsIr($open,20),50)",
    "Sign(TsSkew(Sub(30.0,Rank($close)),10))",
    "Mul(Less(10.0,Add(Sub(Sub(Rank($volume),-30.0),2.0),0.5)),TsMinDiff(Div(Greater($close,0.01),$high),10))",
    "Pow(TsWMA(Pow(5.0,Sub(Log(TsIr($open,40)),-10.0)),40),-2.0)",
    "TsKurt($volume,40)",
    "TsKurt(TsMaxDiff($open,10),50)",
    # PPO / AlphaGen; data/ppo_logs/pool_20/ppo_csi300_20_0-20260629104221/ppo_csi300_20_0_20260629104221/90112_steps_pool.json
    "Add(-0.5,TsMaxDiff($vwap,50))",
    "TsDelta(Add(Pow(2.0,Pow(TsSum($close,30),-1.0)),-2.0),50)",
    "Pow(0.01,Div($vwap,TsWMA($low,20)))",
    "Pow(Greater(1.0,SLog1p($open)),TsDiv($low,20))",
    "Add(-5.0,Log(Log($volume)))",
    "Mul(2.0,TsIr($volume,50))",
    "TsDiv($close,30)",
    "SLog1p(Div(Mul(-5.0,Log(SLog1p($low))),SLog1p($close)))",
    "Pow(0.01,Div(TsIr($high,50),30.0))",
    "SLog1p(Add(5.0,TsDiv($open,50)))",
    "TsMaxDiff(Log($close),30)",
    "Inv(Add(1.0,TsSum(TsMinDiff($close,10),10)))",
    "Div(10.0,$low)",
    "TsCorr(Add(-10.0,$vwap),$volume,30)",
    "Div(TsStd(Log(Add(5.0,$vwap)),10),-5.0)",
    "TsEMA($low,30)",
    "Pow(30.0,Div($close,$vwap))",
    "TsStd(Sub(0.5,Div($open,$high)),20)",
    "Mul(2.0,Ref(Log($low),50))",
    "TsCorr($vwap,Div($close,-0.01),30)",
    # LLM / AlphaKnowledge; data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/pool_1.json
    "Div(Sub(Less($open,$close),$low),$open)",
    "Div(Sub(Less($open,$close),$low),Add(Sub($high,$low),1e-06))",
    "Div(Sub($close,$open),Add(Sub($high,$low),1e-06))",
    "Div(Sub(Sub(Mul(2.0,$close),$high),$low),Add(Sub($high,$low),1e-06))",
    "Div(Sub($high,GetGreater($open,$close)),Add(Sub($high,$low),1e-06))",
    "Div(TsMean(GetGreater(Sub($high,$low),GetGreater(Abs(Sub($high,Ref($close,1))),Abs(Sub($low,Ref($close,1))))),10),$close)",
    "Sub(TsMean(Greater($close,Ref($close,1)),10),TsMean(Less($close,Ref($close,1)),10))",
    "TsMean(Less($close,Ref($close,1)),10)",
    "TsRank($close,10)",
    "Div(Sub($close,TsMin($low,10)),Add(Sub(TsMax($high,10),TsMin($low,10)),1e-06))",
    "Div(Sub(TsSum(GetGreater(Sub($close,Ref($close,1)),0.0),10),TsSum(GetGreater(Sub(Ref($close,1),$close),0.0),10)),Add(TsSum(Abs(Sub($close,Ref($close,1))),10),1e-06))",
    "Div(TsStd(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),10),Add(TsMean(Mul(Abs(Sub(Div($close,Ref($close,1)),1.0)),$volume),10),1e-06))",
    "Div(TsMean($volume,10),Add($volume,1e-06))",
    "Div(Sub(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),10),TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),10)),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),10),1e-06))",
    "Div(TsSum(GetGreater(Sub($volume,Ref($volume,1)),0.0),10),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),10),1e-06))",
    "Div(TsSum(GetGreater(Sub(Ref($volume,1),$volume),0.0),10),Add(TsSum(Abs(Sub($volume,Ref($volume,1))),10),1e-06))",
    "TsMean(Div(Sub($vwap,$close),$vwap),5)",
    "TsSkew(Div(Sub($close,$open),$open),30)",
    "Div(TsStd($volume,10),Add(TsMean($volume,10),0.0001))",
    "TsSkew(Div(Sub($volume,Ref($volume,1)),Add(Ref($volume,1),0.0001)),10)",
    "TsMean(Div(Sub($close,$open),$open),10)",
]


def load_data(args, split, log):
    # Hard-coded split windows are deliberate: each split has its own
    # boundary so a factor can't accidentally leak across train/valid/test
    # boundaries (the alpha_pool.utility() also depends on this).
    intervals = {"train": ("2010-01-01", "2021-12-31"),
                 "valid": ("2022-01-01", "2022-12-31"),
                 "test": ("2023-01-01", "2026-04-30")}
    start, end = intervals[split]
    calendar = pd.DatetimeIndex(pd.read_csv(Path(args.qlib_path) / "calendars/day.txt", header=None)[0])
    left, right = calendar.searchsorted(pd.Timestamp(start)), calendar.searchsorted(pd.Timestamp(end), side="right")
    # Refuse to run if the calendar doesn't extend back far enough for the
    # longest rolling window, or forward enough for the test horizon — the
    # expression would index out of bounds and the failure would be silent.
    if (not calendar.is_monotonic_increasing or calendar.has_duplicates or
            left < args.max_backtrack or right <= left or calendar[-1] < pd.Timestamp(end)):
        raise ValueError(f"Calendar does not cover {split} and its history: {start}..{end}")
    data = StockData(instrument=args.instrument, start_time=str(calendar[left].date()),
                     end_time=str(calendar[right - 1].date()), max_backtrack_days=args.max_backtrack,
                     max_future_days=0, device=torch.device(args.device), qlib_path=args.qlib_path)
    expected = calendar[left - args.max_backtrack:right]
    if not pd.DatetimeIndex(data._dates).equals(expected) or data.n_stocks < 2 or data.n_days <= args.horizon + 1:
        raise ValueError(f"Unexpected dates/shape in {split} data")
    close = Feature(FeatureType.CLOSE).evaluate(data)
    # Forward horizon-day return as the label: IC is measured against this.
    target = close[args.horizon:] / close[:-args.horizon] - 1
    # Mask out days where either endpoint price is non-positive (corporate
    # actions, delistings, etc.) or the ratio is non-finite — they would
    # poison Spearman otherwise.
    usable = (close[args.horizon:] > 0) & (close[:-args.horizon] > 0) & torch.isfinite(target)
    target = target.masked_fill(~usable, torch.nan)
    if (usable.sum(1) >= 2).sum().item() < 2:
        raise ValueError(f"Insufficient valid labels in {split}")
    # Drop the cached pandas frame so the GPU tensor is the only live copy;
    # we won't need it again and it keeps memory bounded across splits.
    data.df_bak = None
    log("data_loaded", split=split, dates=[str(calendar[left].date()), str(calendar[right - args.horizon - 1].date())],
        days=len(target), stocks=data.n_stocks, horizon=args.horizon)
    return data, target


def train(args):
    # Two-phase run: search on the train split, then re-select the same
    # expressions on the validation split. The reserve_seconds buffer keeps
    # the trainer from spending the entire budget and leaving nothing for
    # the validation phase.
    started = time.monotonic()
    wall_start = datetime.now()
    log_dir = Path("data/cf_logs") / f"{wall_start:%Y%m%d_%H%M%S_%f}_{args.instrument}_{args.seed}"
    log_dir.mkdir(parents=True)
    log = make_logger(log_dir)
    save_json(log_dir / "args.json", vars(args))
    log("run_start", initial_count=len(INITIAL_EXPRS), model=os.environ["OPENAI_MODEL_NAME"])
    deadline = started + args.max_seconds
    data, target = load_data(args, "train", log)
    pool = AlphaCFPool(data, target, args, partial(log, split="train"),
                       deadline=deadline - args.reserve_seconds, max_evals=args.max_evals)
    trainer = AlphaCFTrainer(pool, args, log_dir)
    expressions = trainer.train(INITIAL_EXPRS)
    save_json(log_dir / "search_pool.json", dict(**pool.to_dict(), args=vars(args)))
    # Explicitly free the train-phase GPU state before allocating the
    # validation pool — `torch.cuda.empty_cache()` releases blocks back to
    # the driver but only after the Python references are gone.
    del trainer, pool, data, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data, target = load_data(args, "valid", log)
    pool = AlphaCFPool(data, target, args, partial(log, split="valid"), deadline=deadline)
    # Re-select from the search pool on the validation window. The factor
    # strings themselves are unchanged; only the cache and selection log
    # are rebuilt from scratch.
    pool.exprs = pool.select(expressions, args.final_size)
    pool.keep(pool.exprs)
    utility = pool.utility(pool.exprs)
    save_json(log_dir / "final.json", dict(**pool.to_dict(), args=vars(args), utility=utility, split="valid"))
    log("finished", size=len(pool.exprs), utility=utility, seconds=time.monotonic() - started)
    print(f"Final factors: {log_dir / 'final.json'}", flush=True)

    # run_adaptive_combination.py
    from run_adaptive_combination import run as run_combo
    combo_args = argparse.Namespace(
        expressions_file=str(log_dir / "final.json"),
        instruments=args.instrument, train_end_year=2021,
        threshold_ric=0.015, threshold_ricir=0.15,
        seed=args.seed, cuda=args.cuda, n_factors=20,
        chunk_size=400, window="inf", label_days=args.horizon,
        use_weights=False, corr_threshold=0.95,
        ridge_alpha=1e-6, use_vif=False, linear_dep_tol=1e-10,
    )
    started_combo = time.monotonic()
    try:
        run_combo(combo_args)
        returncode = 0
    except Exception as exc:
        returncode = 1
        log("combo_failed", error=type(exc).__name__, message=str(exc))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json(log_dir / "combo.json",
              dict(returncode=returncode,
                   ret_s=str(log_dir / "ret_s.npy"),
                   seconds=time.monotonic() - started_combo))
    log("combo_done", returncode=returncode,
        seconds=time.monotonic() - started_combo)
    wall_end = datetime.now()
    hours, rem = divmod((wall_end - wall_start).total_seconds(), 3600)
    minutes = int(rem // 60)
    def _fmt(dt):
        return f"{dt.year}-{dt.month}-{dt.day} {dt.hour:02d}:{dt.minute:02d}"
    print(f"运行时间:{_fmt(wall_start)} - {_fmt(wall_end)} 用时{int(hours)}时{minutes}分", flush=True)


def evaluate_test(args):
    # Test-only entry point: re-evaluate the *exact* factors saved by the
    # validation phase, with no reselection. The args namespace is rebuilt
    # from final.json so a model trained weeks ago can still be replayed,
    # but the device and time budget come from the current invocation.
    source = Path(args.test_only).resolve()
    saved = json.loads(source.read_text(encoding="utf-8"))
    if saved.get("split") != "valid" or len(saved["exprs"]) != saved["args"]["final_size"]:
        raise ValueError("--test-only requires a completed final.json from Validation")
    device, seconds = args.device, args.max_seconds
    restored = {**vars(args), **saved["args"]}
    args = argparse.Namespace(**restored)
    args.device, args.max_seconds = device, seconds
    log = make_logger(source.parent)
    expressions = [parse(text, args) for text in saved["exprs"]]
    if len({str(e) for e in expressions}) != len(expressions):
        raise ValueError("Final factors must be unique")
    deadline = time.monotonic() + seconds
    data, target = load_data(args, "test", log)
    pool = AlphaCFPool(data, target, args, partial(log, split="test"), deadline=deadline)
    # Fail fast if any frozen factor is no longer valid (e.g. the parser
    # was changed since the run was made). We deliberately do NOT reselect.
    for expr in expressions:
        if pool.evaluate(expr) is None:
            raise ValueError(f"Frozen test factor is invalid; no reselection performed: {expr}")
    pool.exprs = expressions
    # Pairwise correlation is reported alongside utility so the consumer
    # can tell whether the pool is actually diverse or just a cluster of
    # near-duplicates that happen to look good in aggregate.
    correlations = [pool.correlation(a, b) for i, a in enumerate(expressions) for b in expressions[i + 1:]]
    report = dict(**pool.to_dict(), utility=pool.utility(expressions), split="test", source=str(source),
                  mean_abs_correlation=float(np.mean(correlations)) if correlations else 0.0,
                  max_abs_correlation=max(correlations, default=0.0), args=vars(args))
    save_json(source.parent / "test.json", report)
    log("test_done", size=len(expressions), utility=report["utility"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="csi300")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0, help="GPU index; -1 for CPU")
    parser.add_argument("--qlib-path", default=None)
    parser.add_argument("--test-only", metavar="FINAL_JSON")
    # Integer-valued flags are registered with both dashed and underscored
    # spellings so `--max-nodes` and `--max_nodes` are equivalent — useful
    # when copy-pasting from a notebook vs. a shell history.
    for name, default in {"rounds": 10, "parents": 6, "mechanisms": 3, "offspring": 12,
                          "refine_top_k": 2, "pool_capacity": 60, "final_size": 30,
                          "horizon": 20, "max_nodes": 50, "max_depth": 7, "max_backtrack": 252,
                          "chunk_size": 64, "max_evals": 1500, "max_llm_calls": 100,
                          "max_seconds": 10800, "reserve_seconds": 900, "memory_limit": 24,
                          "min_stocks": 10, "min_valid_days": 60, "max_refine_trials": 25,
                          "max_output_tokens": 8192}.items():
        parser.add_argument(*dict.fromkeys(("--" + name.replace("_", "-"), "--" + name)), type=int, default=default, dest=name)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20, 40, 60])
    parser.add_argument("--constants", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0, 2.0])
    for name, default in {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "cost_weight": 0.2, "temperature": 0.5,
                          "min_coverage": 0.8, "min_day_coverage": 0.8}.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default, dest=name)
    parser.add_argument("--no-cf-evidence", "--no-diagnosis", dest="no_cf_evidence", action="store_true",
                        help="Keep semantic decomposition; skip intervention construction and measurement")
    for flag in ("no-pool-credit", "random-crossover", "no-memory", "no-pool-selection", "no-refinement"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    positive = ("parents", "mechanisms", "offspring", "pool_capacity", "final_size", "horizon",
                "max_nodes", "max_depth", "max_backtrack", "chunk_size", "max_evals", "max_llm_calls", "max_seconds",
                "max_refine_trials", "max_output_tokens")
    if any(getattr(args, name) <= 0 for name in positive) or min(args.rounds, args.refine_top_k, args.memory_limit) < 0:
        parser.error("Counts and budgets must be positive; rounds/refine-top-k/memory-limit may be zero")
    if not 0 <= args.reserve_seconds < args.max_seconds or args.final_size > args.pool_capacity or min(args.windows) < 2:
        parser.error("Invalid time reserve, pool sizes, or window grid")
    if not all(np.isfinite(v) and v >= 0 for v in (args.alpha, args.beta, args.gamma, args.cost_weight, args.temperature)):
        parser.error("Weights and temperature must be finite and nonnegative")
    if args.min_stocks < 2 or args.min_valid_days < 2:
        parser.error("min-stocks and min-valid-days must be at least two")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (args.min_coverage, args.min_day_coverage)):
        parser.error("Coverage thresholds must lie in [0,1]")
    if not all(np.isfinite(v) for v in args.constants):
        parser.error("Constant grid must be finite")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args.device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device {args.device} for {args.instrument} with seed {args.seed}", flush=True)
    # Per-instrument qlib dataset resolution order: CLI flag, then the
    # matching env var, then a hard-coded repo-relative default. This
    # makes the same script run on both CN and US instruments without
    # any code change.
    env_key = "QLIB_PATH_SP500" if args.instrument == "sp500" else "QLIB_PATH_CN"
    default_data = "data/qlib_data/us_data_qlib_latest" if args.instrument == "sp500" else "data/qlib_data/cn_data_rolling"
    args.qlib_path = str(Path(args.qlib_path or os.environ.get(env_key) or default_data).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.test_only:
        evaluate_test(args)
    else:
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"):
            if not os.environ.get(name):
                parser.error(f"Missing {name} in environment/.env")
        # AGENT.md specifies MiniMax-M3 for this project.
        if os.environ["OPENAI_MODEL_NAME"].lower() != "minimax-m3":
            parser.error("This project uses MiniMax-M3; check OPENAI_MODEL_NAME")
        train(args)


if __name__ == "__main__":
    main()
