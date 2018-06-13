import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.size']=12
#plt.rcParams['figure.figsize']=15,20


class ResultParser:

    def __init__(self, base):
        self.base = base
        self.last_figure_index = 0
        data_structure = {}

        for f in os.listdir('configs'):
            if f[0:len(base)] == base:
                run_id = f.split('.')[0]
                data_structure[run_id] = {}  # init empty dict
                for r in os.listdir('results'):
                    if run_id in r:
                        if r[-3:] == 'csv':
                            data_structure[run_id]['agg'] = 'results/' + r
                        elif 'request' in r.split('_'):
                            data_structure[run_id]['req'] = 'results/' + r
                        elif 'taxi' in r.split('_'):
                            data_structure[run_id]['taxi'] = 'results/' + r
        self.data_structure = data_structure

        # human readable legends
        self.algnames = {
            'baseline_random_user_nearest_taxi': 'nearest',
            'baseline_random_user_random_taxi': 'random',
            'levelling2_random_user_nearest_poorest_taxi_w_waiting_limit': 'poorest'
         }
        self.coldict = {
            "avg_trip_avg_price": "avg_income",
            "std_trip_avg_price": "std_income"
        }

        # which aggregate data cols to use
        self.agg_cols = [
            "num_taxis", "matching", "d", "R",
            "avg_trip_avg_price", "std_trip_avg_price",
            "entropy_ratio_online", "avg_ratio_online",
            "avg_ratio_cruising", "avg_request_last_waiting_times"
        ]

        # which fixed parameters we can use, and what are the varying parameters in that case
        self.cases = {
            "fixed_taxis" : "R",
            "fixed_ratio" : "d"
        }

        self.taxi_agg_plot_vars = [
            "avg_trip_avg_price",
            "std_trip_avg_price",
            "entropy_ratio_online",
            "avg_ratio_cruising"
        ]

        self.request_agg_plot_vars = [
            'avg_request_completed',
            'avg_request_last_waiting_times'
        ]

    def extract_conf(self, run_id):
        """
        Get configuration for a run_id.
        """

        try:
            conf = pd.Series(json.load(open('configs/' + run_id + '.conf')))
        except FileNotFoundError:
            print("No config file for run_id "+run_id+" found in ResultParser with base "+self.base+"!")
            return

        return conf

    def extract_aggregate_metrics(self, run_id):
        try:
            res_agg = pd.read_csv(self.data_structure[run_id]['agg'], header=0, index_col=0)
            res_agg = res_agg.loc[res_agg.shape[0]-1]
        except KeyError:
            print("No aggregate csv file for run_id " + run_id + " found in ResultParser with base " + self.base + "!")
            return

        return res_agg

    def extract_distribution(self, run_id, mode='taxi'):

        try:
            res = pd.Series(json.load(open(self.data_structure[run_id][mode])))
        except KeyError:
            print("No distribution file for run_id " + run_id + " found in ResultParser with base " + self.base +
                  " and mode "+mode+"!\n",self.data_structure[run_id][mode])
            res = None

        return res

    def collect_aggregate_data(self, run_id):
        conf = self.extract_conf(run_id)
        res_agg = self.extract_aggregate_metrics(run_id)

        if (conf is not None) and (res_agg is not None):
            conf['R'] = round(conf['request_rate'] * res_agg['avg_request_lengths'] / conf['num_taxis'], 1)
            conf['d'] = round(np.sqrt(conf['n'] * conf['m'] * 1e4 / conf['num_taxis']), 0)
            agg = pd.concat([conf, res_agg])
        else:
            agg = None

        return agg

    def prepare_all_data(self, case='fixed_taxis'):
        try:
            var = self.cases[case]
        except KeyError:
            print("No such case (" + case + ") is known!")
            return

        l = [[
            self.extract_aggregate_metrics(run_id),
            self.extract_distribution(run_id, "taxi"),
            self.extract_distribution(run_id, "req"),
            self.extract_conf(run_id)] for run_id in self.data_structure if case in run_id]
        l = list(map(pd.concat, filter(lambda x: np.all([elem is not None for elem in x]), l)))

        for elem in l:
            elem['R'] = round(elem['request_rate'] * elem['avg_request_lengths'] / elem['num_taxis'], 1)
            elem['d'] = round(np.sqrt(elem['n'] * elem['m'] * 1e4 / elem['num_taxis']), 0)

        df = pd.concat(l, axis=1).T
        df['matching'] = df['matching'].map(self.algnames)
        df.columns = list(map(lambda x: self.coldict.get(x, x), df.columns))

        return df

    def create_agg_plot(self, case='fixed_taxis'):
        df = self.prepare_all_data(case)

        if df is not None:
            fig, ax = plt.subplots(num=self.last_figure_index, nrows=2, ncols=2, figsize=(15, 15))
            plt.subplots_adjust(hspace=.3)
            self.last_figure_index += 1
            for i, c in enumerate(self.taxi_agg_plot_vars):
                data = pd.pivot_table(
                    df,
                    index='matching',
                    columns=self.cases[case],
                    values=self.coldict.get(c, c),
                    aggfunc=lambda x: x
                )
                data.plot(
                    kind='bar',
                    ax=ax[int(i / 2), i % 2],
                    rot=0,
                    legend=False
                )
                if i == 0:
                    legend_handles, legend_labels = ax[int(i / 2), i % 2].get_legend_handles_labels()
                ax[int(i / 2), i % 2].set_ylabel(self.coldict.get(c, c))
                ax[int(i / 2), i % 2].ticklabel_format(axis='y',style="sci", scilimits=(-2, 2))
                ax[int(i / 2), i % 2].grid()
            fig.legend(legend_handles, legend_labels, title=self.cases[case], loc="upper center", ncol=len(legend_labels))
            plt.show()

            fig2, ax2 = plt.subplots(num=self.last_figure_index, nrows=1, ncols=2, figsize=(15, 7.5))
            plt.subplots_adjust(wspace=.3)
            self.last_figure_index+=1
            for i, c in enumerate(self.request_agg_plot_vars):
                data = pd.pivot_table(
                    df,
                    index='matching',
                    columns=self.cases[case],
                    values=self.coldict.get(c, c),
                    aggfunc=lambda x: x
                )
                data.plot(
                    kind='bar',
                    ax=ax2[i % 2],
                    rot=0,
                    legend=False
                )
                ax2[i % 2].set_ylabel(self.coldict.get(c, c))
                ax2[i % 2].ticklabel_format(style="sci", axis='y', scilimits=(-2, 2))
                ax2[i % 2].grid()
                if i == 0:
                    legend_handles, legend_labels = ax2[i % 2].get_legend_handles_labels()
            fig2.legend(legend_handles, legend_labels, title=self.cases[case], loc="upper center", ncol=len(legend_labels))
            plt.show()

        else:
            print("No data to plot.")
            return

    def create_distr_plot(self, case='fixed_taxis', col='trip_avg_price'):
        try:
            var = self.cases[case]
        except KeyError:
            print("No such case (" + case + ") is known!")
            return

        df = self.prepare_all_data(case)
        if df is not None:
            data = pd.pivot_table(df, index=var, columns='matching', values=col, aggfunc=lambda x: x)

            data['min'] = data.apply(lambda row: min([min(row[c]) for c in ['nearest', 'poorest', 'random']]), axis=1)
            data['max'] = data.apply(lambda row: max([max(row[c]) for c in ['nearest', 'poorest', 'random']]), axis=1)

            nr = int((len(data.index)-0.1)/2)+1
            fig, ax = plt.subplots(nrows=nr, ncols=2, figsize=(15, 7.5*nr))
            plt.subplots_adjust(hspace=.3)
            legend_handles = []
            legend_labels = []
            for i, var in enumerate(data.index):
                for c in ['nearest', 'poorest', 'random']:
                    y, x = np.histogram(
                        data.loc[var][c],
                        range=(data.loc[var]['min'], data.loc[var]['max']),
                        bins=100,
                        density=True
                    )
                    h = ax[int(i / 2), i % 2].fill_between(x[1:], 0, y, label=c, alpha=0.7)
                    if i == 0:
                        legend_handles.append(h)
                        legend_labels.append(c)
                ax[int(i / 2), i % 2].set_ylabel(r'$p($'+col+'$)$')
                ax[int(i / 2), i % 2].set_xlabel('$'+self.cases[case]+'=%.2f$' % var)
                ax[int(i / 2), i % 2].ticklabel_format(style="sci",scilimits=(-2, 2))
                ax[int(i / 2), i % 2].grid()
                if len(data.index)%2 == 1 and i == len(data.index)-1:
                    ax[int(i / 2), 1].set_axis_off()
            fig.legend(legend_handles, legend_labels, title="algorithm", loc="upper center", ncol=len(legend_labels))
            plt.show()
        else:
            print("No data to plot.")
            return


if __name__ == "__main__":
    base = sys.argv[1]
    rp = ResultParser(base)

    rp.create_agg_plot('fixed_ratio')
    rp.create_agg_plot('fixed_taxis')
    rp.create_distr_plot('fixed_ratio')
    rp.create_distr_plot('fixed_taxis')