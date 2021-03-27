#!env/bin/python
import argparse
from datetime import datetime
import os
import pickle
import sys

from attrdict import AttrDict #type:ignore
import numpy as np #type:ignore
import random as r


import _arguments
import build
import clustering
import distances
import firefly
import log
import permutation
import route
import csv

from typing import Callable, Dict, List, Optional, Tuple
Node = Tuple[int, int]
Value = route.Plan
#しきり数(チェックポイント数どこにあるんかわからん)
item = 31
#遺伝子数
idensisuu = 1
#GAのループ数
galoop = 0
#データ収集用のログファイル名
newfilename = 'out/kekka.csv'

def main():#初めに読まれるとこ

    try:
        args = _arguments.parse()
    except RuntimeError as e:
        print('Argument Error: {}'.format(e))
        return -1

    pathdata = route.PathData(args.input)

    #初期値を乱数で生成、代入 
    e = [r.choices([0,1], k=item) for _ in range(idensisuu)]
    e[0] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    args.n_drones = (sum(x==1 for x in e[0]) + 1) #argsにe[0]のドローン数を代入してしまう

    calc_value = get_calc_value(args, pathdata=pathdata, e=e[0]) #ここに順列を渡すと、評価値を返してくれる ここでeに具体的な値を渡さないとだめ
    init_p_perms_by_seed:Dict[int, List[build.PatternedPermutation]] = {}
    states_by_seed  :Dict[int, Dict[int, AttrDict]] = {}
    bests_by_seed   :Dict[int, List[Value]] = {}
    variants_by_seed:Dict[int, List[int]] = {}

    start_seed = args.seed
    start_init_seed = args.init_seed

    for seed in range(start_seed, start_seed + args.n_run):

        args.seed = seed
        args.init_seed = (seed - start_seed) + start_init_seed
        args.output_filename = '{}/{}.txt'.format(args.output, datetime.now().strftime("%Y%m%d_%H%M%S_%f")) if args.n_run > 1 else '{}.txt'.format(args.output)

        init_p_perms_by_seed[args.seed], states_by_seed[args.seed], bests_by_seed[args.seed], variants_by_seed[args.seed] \
             = run(args, pathdata = pathdata, calc_value = calc_value, e=e) #run関数を呼んでる


    if not args.no_binary_output:
        out_bin = AttrDict()
        out_bin.args = args
        out_bin.init_p_perms_by_seed = init_p_perms_by_seed
        out_bin.states_by_seed = states_by_seed
        out_bin.final_states_by_seed = {seed:list(states.items())[-1][1] for seed, states in states_by_seed.items()}
        out_bin.bests_by_seed = bests_by_seed
        out_bin.variants_by_seed = variants_by_seed

        path = args.binary_output if args.binary_output is not None else args.output + '.pickle'
        log.prepare_directory(path)
        with open(path, mode='wb') as f:
            pickle.dump(out_bin, file = f)



def run(args, *,
    pathdata  : route.PathData,
    calc_value : Callable,
    e : List[List[int]],                    # ドローン数と割り当て方が入ってる
) -> Dict[int, AttrDict]:

    logfile = make_logfile_writer(args)

    # 出力の基本的な情報
    logfile.write('#Program\tRoute Planner')
    logfile.write('#Args\t{}'.format(vars(args)))

    init_p_perms:List[build.PatternedPermutation] = init(args, pathdata = pathdata)
    init_indivs = [p_perm.nodes for p_perm in init_p_perms]
    val_of = list(map(calc_value, init_indivs)) # ここでe[0]の評価値算出される

    if not args.no_init_output:
        logfile.write('#Initialization')
        output_values(args, init_p_perms, val_of, logfile)
        logfile.write('#END').flush()

    if not args.init_only:
        logfile.write('#Iterations')
        states, bests_by_update, variants_by_update = optimize(args, logfile, nodes=pathdata.nodes, calc_value=calc_value, init_indivs=init_indivs, init_val_of=val_of, e=e, pathdata = pathdata)
        logfile.write('#END') # 上でoptimize関数を呼んでる
    
    logfile.write('#EOF').flush()

    return init_p_perms, states, bests_by_update, variants_by_update



def optimize(
    args,
    logfile  : log.FileWriter,
    *,
    nodes    : List[Node],
    calc_value: Callable,
    init_indivs : List[List[Node]],
    init_val_of : List[Value],
    e               : List[List[int]],                    # ドローン数と割り当て方が入ってる
    pathdata  : route.PathData,
) -> Dict[int, AttrDict]:

    np.random.seed(seed = args.seed)

    states:Dict[int, AttrDict] = {}

    bests_by_update:List[Value] = []
    variants_by_update:List[int] = []

    nexte = [[0] * item for _ in range(idensisuu)]
    p = [0] * idensisuu

    for i in range(galoop):
        logfile.write('<GA'+str(i)+'回目>').flush()
        for j in range(idensisuu):
            logfile.write('<e['+str(j)+']の経路探索>').flush()
            args.n_drones = (sum(x==1 for x in e[j]) + 1)
            calc_value = get_calc_value(args, pathdata=pathdata, e=e[j]) # 遺伝子更新
        # Run firefly algorithm
            for state in firefly.run(
                nodes            = nodes,
                init_indivs      = init_indivs,
                init_val_of      = init_val_of,
                calc_value       = calc_value,
                continue_coef    = make_continue_coef(args),
                gamma            = args.gamma,
                alpha            = args.alpha,
                blocked_alpha    = args.blocked_alpha,
                skip_check       = args.skip_check,
                use_jordan_alpha = args.use_jordan_alpha,
                bests_out        = bests_by_update,
                variants_out     = variants_by_update,
                e                = e[j],
            ):  
                states[state.itr] = state
                
                # if state.itr == state.best_itr: # ログ出まくるから経路は最後の一回だけ出力
                #     logfile.write(args.format_itr.format(
                #         t    = state.itr,
                #         nup  = state.n_updates,
                #         nbup = state.n_best_updates,
                #         v    = state.best_plan.value, #評価値
                #         sv   = state.best_plan.average_safety, #不確かさ
                #         dv   = state.best_plan.total_distance, #経路長
                #         log  = state.best_plan.text,
                #     )).flush()

                if args.show_progress and state.itr % 10 == 0:
                    print('.', file=sys.stderr, end='')
                    sys.stderr.flush()
            #ホタルアルゴリズムここまで
            p[j] = state.best_plan.value
        
        for j in range(idensisuu): # 今ある遺伝子をlog出力
            logfile.write('e['+str(j)+']:'+str(e[j])+' 評価値:'+str(p[j])).flush()
        

        if i == 0 and j == 0:
            with open(newfilename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(p)
        else:
            with open(newfilename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(p)
        
        p, e, nexte = choice(p, e, nexte)
        nexte = generation(nexte)
        nexte = mutation(nexte)
        e = inherit(e,nexte)
    
    logfile.write('<GA'+str(galoop)+'回目>').flush()
    for i in range(idensisuu):
        logfile.write('<e['+str(i)+']の経路探索>').flush()
        args.n_drones = (sum(x==1 for x in e[i]) + 1)
        calc_value = get_calc_value(args, pathdata=pathdata, e=e[i]) # 遺伝子更新
        for state in firefly.run(
                nodes            = nodes,
                init_indivs      = init_indivs,
                init_val_of      = init_val_of,
                calc_value       = calc_value,
                continue_coef    = make_continue_coef(args),
                gamma            = args.gamma,
                alpha            = args.alpha,
                blocked_alpha    = args.blocked_alpha,
                skip_check       = args.skip_check,
                use_jordan_alpha = args.use_jordan_alpha,
                bests_out        = bests_by_update,
                variants_out     = variants_by_update,
                e                = e[i],
            ):  
                states[state.itr] = state
                
                if state.itr == state.best_itr:
                    
                    logfile.write(args.format_itr.format(
                        t    = state.itr,
                        nup  = state.n_updates,
                        nbup = state.n_best_updates,
                        v    = state.best_plan.value, #評価値
                        sv   = state.best_plan.average_safety, #不確かさ
                        dv   = state.best_plan.total_distance, #経路長
                        log  = state.best_plan.text,
                    )).flush()

                if args.show_progress and state.itr % 10 == 0:
                    print('.', file=sys.stderr, end='')
                    sys.stderr.flush()
        p[i] = state.best_plan.value

    logfile.write(args.format_terminate.format(t = state.itr, nup = state.n_updates))
    
    for i in range(idensisuu): # 最終的に残った遺伝子をlog出力
        logfile.write('e['+str(i)+']:'+str(e[i])+' 評価値:'+str(p[i])).flush()
    logfile.write('<最も良い遺伝子と評価値>').flush()
    logfile.write('e['+str(p.index(min(p)))+']:'+str(e[p.index(min(p))])+' 評価値:'+str(min(p))).flush()

    with open(newfilename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(p)
        
    if args.show_progress:
        print('', file=sys.stderr)

    return states, bests_by_update, variants_by_update



def init(args, *, pathdata:route.PathData) -> List[build.PatternedPermutation]:

    np.random.seed(seed = args.init_seed)

    bld_dist = distances.get_multiple_func(args.init_bld_dists, pathdata = pathdata)
    cls_dist = distances.get_multiple_func(args.init_cls_dists, pathdata = pathdata)

    n_random = round(args.init_random_rate * args.n_indiv)
    n_special = args.n_indiv - n_random

    if n_random < 0 or n_special < 0 or n_random + n_special != args.n_indiv:
        raise RuntimeError('Invalid number of individuals.')

    init_p_perms:List[build.PatternedPermutation] = []

    # Random generation
    for _ in range(n_random):
        init_p_perms.append(build.build_randomly(pathdata.nodes))

    # Cluster-patterned generation
    if n_special:
        clusters_nodes = clustering.get_function(method = args.init_cls_method, nodes = pathdata.nodes, n_cluster = args.init_n_cls, dist = cls_dist, dist_compare = distances.compare_multiple)()

        if args.init_bld_method == 'rg':
            builder = build.Builder(methods_func_dof = {
                'R': (lambda nodes: build.build_randomly(nodes), 1),
                'G': (lambda nodes: build.build_greedy(nodes, bld_dist, distances.compare_multiple, nn_n_random=args.init_greedy_rnum, start_node=pathdata.home_poses[0]), 0),
            }, clusters_nodes = clusters_nodes)
            init_p_perms.extend(builder.build_with_dof(n_special))

        elif args.init_bld_method == 'r':
            for _ in range(n_special):
                init_p_perms.append(build.chain_patterned_permutations([build.build_randomly(c_nodes) for c_nodes in clusters_nodes]))

        else:
            raise RuntimeError('Unknown initialzation building method.')


    # Validation
    if not all([permutation.is_valid(p_perm.nodes, pathdata.nodes) for p_perm in init_p_perms]):
        raise RuntimeError('Invalid individuals.')

    return init_p_perms



def get_calc_value(args, *, pathdata:route.PathData, e:List[int]) -> Callable:

    plan_generator = route.PlanGenerator(
        pathdata = pathdata,
        drone_prop = route.DroneProperty(pathdata),
        n_drones = args.n_drones,
        drone_n_cp = args.drone_n_cp,
        safety_weight = args.safety_weight,
        distance_weight = args.distance_weight,
        e = e,
    )

    return lambda perm : plan_generator.make([perm], e) # lambda 引数: 返り値



def output_values(args, init_p_perms:List[build.PatternedPermutation], val_of:list, logfile:log.FileWriter):

    for i, (p_perm, val) in enumerate(zip(init_p_perms, val_of)):
        logfile.write(args.format_init.format(
            i    = i,
            v    = val.value,
            sv   = val.average_safety,
            dv   = val.total_distance,
            log  = val.text,
            pat  = pattern_to_str(p_perm.pattern)
        ))


def pattern_to_str(pattern):
    if isinstance(pattern, tuple):
        return '(' + ' '.join(map(pattern_to_str, pattern)) + ')'

    if isinstance(pattern, str):
        return pattern
    
    raise RuntimeError('Invalid pattern type.')


def make_continue_coef(args): # falseを返すとホタルのループが終了する

    if args.n_updates: # デフォルトはオフ、繰り返し上限数を設定することもできる
        return lambda idv: idv.n_updates <= args.n_updates

    if args.n_itr_steady: #デフォルトはオフ
        check_steady = lambda idv: (idv.itr - idv.best_itr) < args.n_itr_steady
        if args.n_min_iterate:
            if args.n_max_iterate:
                if args.n_min_iterate > args.n_max_iterate:
                    raise RuntimeError('Maximum iteration is smaller than minimum iteration.')
                return lambda idv: idv.itr <= args.n_min_iterate or (idv.itr <= args.n_max_iterate and check_steady(idv))
            return lambda idv: idv.itr <= args.n_min_iterate or check_steady(idv)
        if args.n_max_iterate:
            return lambda idv: idv.itr <= args.n_max_iterate and check_steady(idv)
        return lambda idv: check_steady(idv)
    else:
        if args.n_min_iterate:
            return lambda idv: idv.itr <= args.n_min_iterate
        if args.n_max_iterate:#argsで1000に設定
            return lambda idv: idv.itr <= args.n_max_iterate # ホタルが1000回回ったらホタル終了。
    
    raise RuntimeError('All of minimum and maximum and steady iterations are not specified.')



def make_logfile_writer(args): # 1個めのログ用のfilewriter

    if args.no_log_output: # ログファイル作らん時にfalseにすればいいやつ
        if args.stdout: return log.FileWriter(outobj=sys.stdout)
        return log.FileWriter(no_out=True)

    if args.stdout: return log.FileWriter(filepath=args.output_filename, outobj=sys.stdout)
    return log.FileWriter(filepath=args.output_filename)

#選択関数
def choice(p, e, nexte):
    #評価値で昇順ソート
    p, e = \
        list(map(list, zip(*sorted(zip(p, e), key=lambda x:x[0], reverse=False))))
    
    # rank = [2, 2, 1, 1, 1, 1, 1, 1, 0, 0] # 遺伝子数が10個の場合
    # rank = [2, 1, 1, 1, 0] # 遺伝子が5個の場合
    # rank = [2, 1, 0] # 遺伝子が3個の場合
    rank = [1] #遺伝子が1個の場合

    #ランキング選択：e の先頭から rank リストに従って次世代へ移行
    nexte = [e[i].copy() for i in range(len(rank)) for _ in range(rank[i])]

    return p, e, nexte


# ! 次世代を操作しているの違和感
#2点交叉
def generation(nexte):
    # ! 親が残らない
    # * エリート一体は交叉なし変異なし
    for i in range(1,len(nexte)-1,2): # ! len(nexte) == 偶数時最後の個体交叉せず
        crossrate = r.randint(0,99)
        if(crossrate < 95):
            # * nexte[i], nexte[i+1] の cross_l:cross_r 区間を交換
            cross_l, cross_r = sorted(r.sample(range(item+1), 2))
            nexte[i][cross_l:cross_r], nexte[i+1][cross_l:cross_r] \
                = nexte[i+1][cross_l:cross_r], nexte[i][cross_l:cross_r]
    
    return nexte

def mutation(nexte):
    #突然変異
    for i in range(1, idensisuu):
        mutantrate = r.randint(0,99)
        # * 5 は少ない
        if(mutantrate < 50):
            m = r.randint(0,item-1)
            nexte[i][m] = (nexte[i][m] + 1) % 2
    
    return nexte


#次世代を現世代へ
def inherit(e,nexte):
    e = nexte.copy()

    return e


if __name__ == '__main__':
    main()
