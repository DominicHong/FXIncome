import numpy as np
import multiprocessing as mp
import pandas as pd
import os
import plotly.express as px
from itertools import repeat


def compare_price(t, dirty=True):
    results = []
    samples = []
    col_names = ['dp', 'dr', 'dt', 'c', 'r1', 'r2', 't1', 't2', 'p1', 'p2']
    c = 2.5
    for r in np.linspace(0.02, 0.03, 20):
        # for c in np.linspace(1, 10, 10):
            for dt in np.linspace(0.003, 1, 10):
                t2 = t - dt
                if t2 < 0.003:
                    continue
                for dr in np.linspace(0.001, 0.02, 20):
                    r2 = r - dr
                    if r2 < 0:
                        continue
                    if dirty:
                        p1 = (100 + c) / (1 + r * t)
                        p2 = (100 + c) / (1 + r2 * t2)
                    else:
                        p1 = (100 + c) / (1 + r * t) - c * (1 - t)
                        p2 = (100 + c) / (1 + r2 * t2) - c * (1 - t2)
                    dp = round(p2 - p1, 4)
                    samples.append((dp, dr, dt, round(c, 2), round(r, 4), round(r2, 4), round(t, 4), round(t2, 4),
                                    round(p1, 4), round(p2, 4)))
                    if dp < 0:
                        results.append((dp, dr, dt, round(c, 2), round(r, 4), round(r2, 4), round(t, 4), round(t2, 4),
                                        round(p1, 4), round(p2, 4)))
                        print(f"Found! p1: {p1:.4f}, p2: {p2:.4f} "
                              f"r1: {r * 100:.4f}, r2: {r2 * 100:.4f}, c: {c:.4f} t1: {t:.4f}, t2: {t2:.4f}")
    df = pd.DataFrame(results, columns=col_names)
    df.to_csv(f'./results/ytm_price-{t}.csv', index=False, encoding='utf-8')
    df = pd.DataFrame(samples, columns=col_names)
    df.to_csv(f'./results/samples-{t}.csv', index=False, encoding='utf-8')


def combine_files():
    results = pd.DataFrame()
    samples = pd.DataFrame()
    for cur_dir, dirs, files in os.walk('./results'):
        for file in files:
            if file == 'ytm_price.csv':
                continue
            elif file == 'samples.csv':
                continue
            df = pd.read_csv(os.path.join(cur_dir, file))
            os.remove(os.path.join(cur_dir, file))
            if file.startswith('ytm'):
                results = results.append(df)
            elif file.startswith('sample'):
                samples = samples.append(df)
    results.to_csv(f'./results/ytm_price.csv', index=False, encoding='utf-8')
    samples.to_csv(f'./results/samples.csv', index=False, encoding='utf-8')
    print(f"results:{len(results)} samples:{len(samples)} hit ratio:{len(results) / len(samples)}")


def concurrent():
    cpus = mp.cpu_count() - 1
    t = np.linspace(0.01, 1, 10)
    dirty = False
    with mp.Pool(cpus) as p:
        p.starmap(compare_price, zip(t, repeat(dirty)))


def plot_3d():
    samples = []
    col_names = ['p', 'r', 't', 'c', 'p_type']
    r_range = np.linspace(0.01, 0.10, 20)
    t_range = np.linspace(0.003, 1, 20)
    c_range = [1, 10]
    for c in c_range:
        for r in r_range:
            for t in t_range:
                dirty_p = (100 + c) / (1 + r * t)
                clean_p = (100 + c) / (1 + r * t) - c * (1 - t)
                samples.append((round(dirty_p, 4), round(r, 4), round(t, 4), c, '全价'))
                samples.append((round(clean_p, 4), round(r, 4), round(t, 4), c, '净价'))
    df = pd.DataFrame(samples, columns=col_names)
    fig = px.scatter_3d(df, x='r', y='t', z='p', color='c', symbol='p_type')
    fig.update_layout(
        title_text='YTM to Price Function, C in [1, 10]',
        title_x=0.5,
        legend_x=0.1
    )
    fig.show()
    df = pd.read_csv('./results/samples.csv')
    df['r_ratio'] = df.dr / df.r1
    df['t_ratio'] = df.dt / (df.t1 - df.dt)
    df = df[df.c == 5]
    fig = px.scatter_3d(df, x='r_ratio', y='t_ratio', z='dp', color='t_ratio',)
    fig.update_layout(
        title_text='Dirty Prices change when rate increases',
        title_x=0.5
    )
    fig.show()
    df.to_csv('./results/samples_detail.csv', index=False)
    # df = pd.read_csv('./results/samples.csv')
    # df.p = [df.p.to_numpy() for x in df.index]
    # fig = px.scatter_matrix(df, dimensions=['p', 'r', 't'])



if __name__ == '__main__':
    concurrent()
    combine_files()
    # plot_3d()
