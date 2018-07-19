import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from generate_configs import avg_length

plt.rcParams['font.size'] = 12

class ResultParser:

    def __init__(self, base):
        print('Initializing ResultParser...')
        self.base = base
        self.last_figure_index = 0
        data_structure = {}
        self.l = None

        for f in os.listdir('configs'):
            if f[0:len(base)] == base:
                if f[len(base)] == '.':
                    print('Reading base conf... configs/'+f)
                    conf = json.load(open('configs/'+f))
                    self.l = avg_length(conf)
                else:
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

        self.agg_plot_vars = {
            'taxi': self.taxi_agg_plot_vars,
            'request': self.request_agg_plot_vars
        }

        print('Done.')

    def extract_conf(self, run_id):
        """
        Get configuration for a run_id.
        """

        try:
            conf = pd.Series(json.load(open('configs/' + run_id + '.conf')))

            if 'R' not in conf:
                conf['R'] = round(conf['request_rate'] * self.l / conf['num_taxis'], 1)
            if 'd' not in conf:
                conf['d'] = round(np.sqrt(conf['n'] * conf['m'] * 1e4 / conf['num_taxis']), 0)
        except FileNotFoundError:
            print("No config file for run_id "+run_id+" found in ResultParser with base "+self.base+"!")
            return

        return conf

    def extract_aggregate_metrics(self, run_id):
        try:
            res_agg = pd.read_csv(self.data_structure[run_id]['agg'], header=0, index_col=0)
            res_agg = res_agg.loc[res_agg.shape[0]-1]
            if 'fixed_ratio' in run_id:
                conf = self.extract_conf(run_id)
                res_agg['entropy_ratio_online'] = res_agg['entropy_ratio_online']/np.log(conf['num_taxis'])
        except KeyError:
            print("No aggregate csv file for run_id " + run_id + " found in ResultParser with base " + self.base + "!")
            return

        return res_agg

    def extract_timelines(self, run_id):
        try:
            res_agg = pd.Series(
                pd.read_csv(
                    self.data_structure[run_id]['agg'], header=0, index_col=0
                ).to_dict(orient='list')
            )
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

        df = pd.concat(l, axis=1).T
        df['matching'] = df['matching'].map(self.algnames)
        df.columns = list(map(lambda x: self.coldict.get(x, x), df.columns))

        return df

    def create_agg_plot(self, case='fixed_taxis'):
        df = self.prepare_all_data(case)

        # taxi metrics
        if df is not None:
            for v in self.agg_plot_vars:
                nr = int((len(self.agg_plot_vars[v]) - 0.1) / 2) + 1
                fig, ax = plt.subplots(num=self.last_figure_index, nrows=nr, ncols=2, figsize=(15, 3.75*nr))
                plt.subplots_adjust(hspace=.3)
                self.last_figure_index += 1

                for i, c in enumerate(self.agg_plot_vars[v]):
                    data = pd.pivot_table(
                        df,
                        index='matching',
                        columns=self.cases[case],
                        values=self.coldict.get(c, c),
                        aggfunc=lambda x: x
                    ).T
                    # the next line is necessary because of a bug in the config generation whch has already been resolved
                    data = data[data['nearest'].map(lambda x: type(x)==float)]
                    if nr>1:
                        current_ax = ax[int(i / 2), i % 2]
                    else:
                        current_ax = ax[i % 2]
                    data.plot(
                        kind='line',
                        style='o--',
                        ax=current_ax,
                        rot=0,
                        legend=False
                    )
                    if i == 0:
                        legend_handles, legend_labels = current_ax.get_legend_handles_labels()
                    current_ax.set_ylabel(self.coldict.get(c, c))
                    current_ax.ticklabel_format(axis='y',style="sci", scilimits=(-2, 2))
                    current_ax.grid()
                fig.legend(
                    legend_handles,
                    legend_labels,
                    title='algorithm',
                    loc="lower center",
                    ncol=len(legend_labels),
                    bbox_to_anchor=(0.5, (-0.01+3.75*nr)/(3.75*nr)),
                    bbox_transform=plt.gcf().transFigure
                )
                plt.savefig(
                    'figs/' + self.base + '_agg_'+v+'_' + case + '.png', dpi=300,
                    bbox_inches='tight'
                )

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

            # the next line is necessary because of a bug in the config generation whch has already been resolved
            data = data[data['nearest'].map(lambda x: len(x) > 0)]

            data['min'] = data.apply(lambda row: min([min(row[c]) for c in ['nearest', 'poorest', 'random']]), axis=1)
            data['max'] = data.apply(lambda row: max([max(row[c]) for c in ['nearest', 'poorest', 'random']]), axis=1)

            nr = int((len(data.index)-0.1)/2)+1
            fig, ax = plt.subplots(num=self.last_figure_index, nrows=nr, ncols=2, figsize=(15, 3.75*nr))
            self.last_figure_index += 1
            plt.subplots_adjust(hspace=.3)

            legend_handles = []
            legend_labels = []
            for i, var in enumerate(data.index):
                if nr > 1:
                    current_ax = ax[int(i / 2), i % 2]
                else:
                    current_ax = ax[i % 2]
                for c in ['nearest', 'poorest', 'random']:
                    y, x = np.histogram(
                        data.loc[var][c],
                        range=(data.loc[var]['min'], data.loc[var]['max']),
                        bins=100,
                        density=True
                    )
                    h = current_ax.fill_between(x[1:], 0, y, label=c, alpha=0.7)
                    if i == 0:
                        legend_handles.append(h)
                        legend_labels.append(c)
                current_ax.set_ylabel(r'$p($'+col+'$)$')
                current_ax.set_xlabel('$'+self.cases[case]+'=%.2f$' % var)
                current_ax.ticklabel_format(style="sci",scilimits=(-2, 2))
                current_ax.grid()
                if len(data.index)%2 == 1 and i == len(data.index)-1:
                    current_ax.set_axis_off()
            fig.legend(
                legend_handles,
                legend_labels,
                title="algorithm",
                loc="lower center",
                ncol=len(legend_labels),
                bbox_to_anchor=(0.5, (-0.01+3.75*nr)/(3.75*nr)),
                bbox_transform=plt.gcf().transFigure
            )
            plt.savefig(
                'figs/'+self.base+'_distr_'+case+'.png',
                bbox_inches='tight'
            )
        else:
            print("No data to plot.")
            return

    def create_timelines(self, case='fixed_taxis'):
        try:
            var = self.cases[case]
        except KeyError:
            print("No such case (" + case + ") is known!")
            return

        l = [[
            self.extract_timelines(run_id),
            self.extract_conf(run_id)] for run_id in self.data_structure if case in run_id]
        l = list(map(pd.concat, filter(lambda x: np.all([elem is not None for elem in x]), l)))

        df = pd.concat(l, axis=1).T
        df['matching'] = df['matching'].map(self.algnames)
        df.columns = list(map(lambda x: self.coldict.get(x, x), df.columns))

        if df is not None:
            for v in self.agg_plot_vars:
                nr = int((len(self.agg_plot_vars[v]) - 0.1) / 2) + 1
                for alg in self.algnames.values():
                    fig, ax = plt.subplots(num=self.last_figure_index, nrows=nr, ncols=2, figsize=(15, 3.75*nr))
                    plt.subplots_adjust(hspace=.3)
                    self.last_figure_index += 1
                    for i, c in enumerate(self.agg_plot_vars[v]):
                        data = pd.pivot_table(
                            df,
                            index='matching',
                            columns=self.cases[case],
                            values=self.coldict.get(c, c),
                            aggfunc=lambda x: x
                        ).T

                        if nr > 1:
                            current_ax = ax[int(i / 2), i % 2]
                        else:
                            current_ax = ax[i % 2]

                        for R in data.index:
                            current_ax.plot(data[alg][R], label=str(R))
                        if i == 0:
                            legend_handles, legend_labels = current_ax.get_legend_handles_labels()
                        current_ax.set_xlabel('Time (1000 tu)')
                        current_ax.set_ylabel(self.coldict.get(c, c))
                        current_ax.ticklabel_format(axis='y', style="sci", scilimits=(-2, 2))
                        current_ax.grid()
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        title=self.cases[case],
                        loc="lower center",
                        ncol=len(legend_labels),
                        bbox_to_anchor=(0.5, (-0.01+3.75*nr)/(3.75*nr)),
                        bbox_transform=plt.gcf().transFigure
                    )
                    plt.savefig(
                        'figs/' + self.base + '_timeline_'+v+'_' + case + '_' + alg + '.png',
                        bbox_inches='tight'
                    )

        else:
            print("No data to plot.")
            return

    def create_map(self, case='fixed_taxis'):
        return


if __name__ == "__main__":
    base = sys.argv[1]
    rp = ResultParser(base)

    rp.create_agg_plot('fixed_ratio')
    rp.create_agg_plot('fixed_taxis')
    rp.create_distr_plot('fixed_ratio')
    rp.create_distr_plot('fixed_taxis')
    rp.create_timelines('fixed_taxis')
    rp.create_timelines('fixed_ratio')

