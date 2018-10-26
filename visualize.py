import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from generate_configs import avg_length # TODO: only OLD = False flag uses it
import re

from ineqpy import atkinson # Atkinson index of inequality

plt.rcParams['font.size'] = 14

# Flags whether we are working with old result files.
OLD = False

class ResultParser:
    """
    This class collects the results belonging to a certain base run_id
    (run_ids are named in a hierarchical fashion as parameterd are being varied).
    It looks for filenames containing the `base` string, and arranges
    the result files into a dictionary structure, where the run_ids are
    the keys, and results belonging to taxis ('taxi'), requests ('req') or
    aggregate metrics ('agg') are form dictionary of filenames
    that lead to the different outputs for the parent run_id.

    e.g. {"run_id1": {"taxi": taxi_file.json, "req": request_file.json,
    "agg": aggregate_metrics.csv}, "run_id2": ...}

    File names are stored with full relative path info, where parent directory
    is root directory of the repository.

    Only those run_ids are listed, for which all three result files could be
    found.

    Some helper class variables are also defined.

    Parameters:
    -----------
    base: str, no default
        Base run_id matching run_ids to collect.
    """

    def __init__(self, base):
        print('Initializing ResultParser...')
        self.base = base
        self.last_figure_index = 0
        data_structure = {}

        for f in os.listdir('configs'):
            if re.match(base, f) is not None:
                run_id = f.split('.')[0]
                data_structure[run_id] = {"counter": 0}  # init empty dict
                for r in os.listdir('results'):
                    if run_id in r:
                        if r[-3:] == 'csv':
                            data_structure[run_id]['agg'] = 'results/' + r
                            data_structure[run_id]["counter"] += 1
                        elif 'request' in r.split('_'):
                            data_structure[run_id]['req'] = 'results/' + r
                            data_structure[run_id]["counter"] += 1
                        elif 'taxi' in r.split('_'):
                            data_structure[run_id]['taxi'] = 'results/' + r
                            data_structure[run_id]["counter"] += 1

        # deleting run_ids with incomplete results
        run_ids = list(data_structure.keys())
        for run_id in run_ids:
            if data_structure[run_id]["counter"] != 3:
                del data_structure[run_id]

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

        self.superfluous = [
            'trip_avg_length',
            'trip_num_completed',
            'trip_std_length'
        ]

        print('Done.')

    def create_data_row(self, run_id):
        """
        This function takes a run_id, and collects all results (aggregated, taxi metrics, request metrics)
        into one single pandas.Series

        Parameters
        ----------
        run_id: str, no default
            run_id to collect results for

        Returns
        -------
        A pandas.Series object containing all possible data for the run_id.

        """
        # extract configuration for run_id
        conf = json.load(open('configs/' + run_id + '.conf'))

        if 'fixed_taxis' in run_id:
            conf['mode']='fixed_taxis'
        elif 'fixed_ratio' in run_id:
            conf['mode'] = 'fixed_ratio'

        # for older config files, TODO: for newer it is always there!
        if OLD == True:
            if 'avg_length' not in conf:
                conf['avg_length'] = avg_length(conf)
            if 'R' not in conf:
                conf['R'] = round(conf['request_rate'] * conf['avg_length'] / conf['num_taxis'], 1)
            if 'd' not in conf:
                conf['d'] = round(np.sqrt(conf['n'] * conf['m'] * 1e4 / conf['num_taxis']), 0)

        # extract aggregate metrics
        # each agg csv contains the same table multiple times under each other
        begin = [] # collecting the line numbers in the file for all data table beginnings
        with open(self.data_structure[run_id]['agg']) as f:
            lc = 0 # line counter
            for line in f:
                if line[0] == ',': # if header, append line number
                    begin.append(lc)
                lc += 1

        if len(begin) == 1: # if there was only one table, the length of the table is the number of lines
            l = lc
        else:
            l = begin[1] - begin[0] # length of one table
        d = []
        for blc in begin: # read each table
            res_agg = pd.read_csv(
                self.data_structure[run_id]['agg'],
                header=0,
                nrows=l - 1,
                skiprows=blc,
                index_col=0
            )
            res_agg = res_agg.loc[res_agg.shape[0] - 1]
            d.append(res_agg)

        # averages and standard deviations
        res_agg = pd.DataFrame(d).mean()
        res_agg_err = pd.DataFrame(d).std()

        # reshaping, renaming
        res_agg = pd.DataFrame([res_agg, res_agg_err]).T
        res_agg.columns = ['mean', 'std']

        l = res_agg.index.tolist()
        res_agg = pd.melt(res_agg)
        res_agg['variable'] = [i for i in l] + [i + '_' + 'std' for i in l]
        res_agg.set_index('variable', inplace=True)
        res_agg = res_agg['value'].to_dict()

        # number of multiple runs
        res_agg['runs'] = len(begin)

        # deleting false entropy calculations, TODO: this step can be omitted for newer files
        if OLD == True:
            for col in res_agg.index:
                if col[0:7] == 'entropy':
                    del res_agg[col]

        mode = 'taxi'

        try:
            with open(self.data_structure[run_id][mode]) as f:
                res = []
                for line in f.readlines():
                    res.append(pd.Series(json.loads(line.strip('\n'))))
                res = pd.DataFrame(res).T
                res_all = res[res.index.map(lambda x: x != 'timestamp')] \
                    .apply(lambda row: [e for l in row.tolist() for e in l], axis=1)

                # Atkinson index of inequality for incomes
                for row in res.index:
                    if row == 'trip_avg_price':
                        e = []
                        for col in res.columns:
                            e.append(atkinson(np.array(res.loc['trip_avg_price'][0])))
                        res_all['atkinson_' + row] = np.mean(e)
                        res_all['atkinson_' + row + '_std'] = np.std(e)
                    elif row!='timestamp':
                        mins = []
                        maxs = []
                        for col in res.columns:
                            mins.append(min(res[col][row]))
                            maxs.append(max(res[col][row]))
                        res_all['min_'+row] = min(mins)
                        res_all['max_' + row] = max(maxs)

                res_all = res_all.to_dict()

                res_agg.update(res_all)
                del res_all
        except json.JSONDecodeError:
            print('Reading error in '+self.data_structure[run_id][mode]+'!')
            return None

        mode = 'req'
        try:
            with open(self.data_structure[run_id][mode]) as f:
                res = []
                for line in f.readlines():
                    res.append(pd.Series(json.loads(line.strip('\n'))))
                res = pd.DataFrame(res).T

                m = res.loc['request_completed'].map(lambda l: np.mean(l))
                res_agg['request_completed'] = np.mean(m)
                res_agg['request_completed_std'] = np.std(m)
        except json.JSONDecodeError:
            print('Reading error in '+self.data_structure[run_id][mode]+'!')
            return None

        for s in self.superfluous:
            del res_agg[s]

        res_agg.update(conf)

        return res_agg

    # TODO: also, now our algorithm dumps results at different timestamps
    # TODO: we should handle it!!!
    def prepare_all_data(self, force=False):
        """
        This function creates a table that contains all the data from all the runs of a certain base.

        The table is written into a csv, if it exists, the default is that the function reads this file.
        If it does not exists, or force is True, then it is generated and (re)written.

        Parameters
        ----------
        force: bool, default False
            Load preparated summary table of results or calculate them again.
        """

        if 'results/' + self.base + '_all.csv' not in os.listdir('results') or force:
            d = []
            for run_id in self.data_structure:
                r = self.create_data_row(run_id)
                if type(r) is not None:
                    d.append(dict(r))
            df = pd.DataFrame.from_dict(d)
            df['matching']=df['matching'].map(self.algnames)
            df.to_csv('results/' + self.base+'_all.csv')
            return df
        else:
            return pd.read_csv('results/' + self.base+'_all.csv', index_col=0, header=0)


class ResultVisualizer:
    """
    This class loads a result parser class with parameter 'base', and then puts
    the results on various kinds of plots that are needed. Each type of plot is
    a new method in this visualizer class.

    Outputs go to the figs folder.

    Parameters
    ----------
    base: str

    """

    def __init__(self, base):
        # loading the available files for the results of the run
        self.rp = ResultParser(base)
        # saving results into a dataframe that is used in the visualizations
        self.df = self.rp.prepare_all_data(force = True)
        self.last_figure_index = 0

        # defining standard colors for the results of the three algorithms
        self.colors = {
            'nearest': 'blue',
            'poorest': 'orange',
            'random': 'green'
        }

    def create_income_plot(self, mode):
        """
        This function plots the average income of taxis and the standard deviation of the income
        distribution at various parameter values of 'd' and 'R'.

        Parameters
        ----------
        mode: str, no default
            either 'fixed_taxis' or 'fixed_ratio', this string determines what to put on the x axis

        Returns
        -------
        None

        """
        # main plot
        fig = plt.figure(num=self.last_figure_index, figsize=(10, 7))
        ax = fig.add_subplot(111)
        self.last_figure_index += 1

        # inset plot for standard deviations
        # they could be represented as relative to the average income!
        inset = plt.axes([.57, .2, .3, .2])

        data = pd.pivot_table(
            self.df[self.df['mode']==mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='avg_trip_avg_price'
        ).T

        data_err = pd.pivot_table(
            self.df[self.df['mode']==mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='std_trip_avg_price'
        ).T

        for col in data.columns:
            ax.plot(data.index,data[col],'o--',label=col,c=self.colors[col],alpha=0.8,markersize=4)
            ax.errorbar(data.index,data[col],yerr=data_err[col],fmt='none',
                         c=self.colors[col],label=None,legend=False,
                         alpha=0.8,capsize=3)
            inset.plot(data.index,data_err[col],'o--',alpha=0.8,c=self.colors[col],markersize=4)

        #ax.set_ylim([0,np.nanmax(data).max()*1.2]) # did not work maybe because of nans, or Nones?
        ax.legend()
        ax.grid()
        ax.set_xlabel(self.rp.cases[mode])
        ax.set_ylabel('Average income')
        inset.set_xlabel(self.rp.cases[mode])
        inset.set_ylabel('Std of income distr.')

        plt.savefig('figs/' + self.rp.base + '_avg_income_'+mode+'.png',dpi=300)

    def create_requests_completed_plot(self, mode):
        """
        This function creates a plot of the completed request ratio.
        
        Parameters
        ----------
        mode: str, no default
            either 'fixed_taxis' or 'fixed_ratio', this string determines what to put on the x axis

        Returns
        -------
        None

        """
        fig = plt.figure(num=self.last_figure_index, figsize=(10, 7))
        ax = fig.add_subplot(111)
        self.last_figure_index += 1

        data = pd.pivot_table(
            self.df[self.df['mode'] == mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='request_completed'
        ).T

        data_err = pd.pivot_table(
            self.df[self.df['mode'] == mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='request_completed_std'
        ).T

        for col in data.columns:
            plt.plot(data.index, data[col], 'o--', label=col, c=self.colors[col], alpha=0.8, markersize=4)
            plt.errorbar(data.index, data[col], yerr=data_err[col], fmt='none',
                         c=self.colors[col], label=None, legend=False,
                         alpha=0.8, capsize=3)
        #plt.ylim([0, np.nanmax(data).max() * 1.2])
        plt.legend()
        plt.grid()
        plt.xlabel(self.rp.cases[mode])
        plt.ylabel('Completed request ratio')

        plt.savefig('figs/' + self.rp.base + '_req_completed_' + mode + '.png', dpi=300)

    def create_entropy_plot(self, mode):
        """
        This function creates the Atkinson index plots.

        Parameters
        ----------
        mode: str, no default
            either 'fixed_taxis' or 'fixed_ratio', this string determines what to put on the x axis

        Returns
        -------
        None

        """
        fig = plt.figure(num=self.last_figure_index, figsize=(10, 7))
        ax = fig.add_subplot(111)
        self.last_figure_index += 1

        print(self.df.columns)

        data = pd.pivot_table(
            self.df[self.df['mode'] == mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='atkinson_trip_avg_price'
        ).T

        data_err = pd.pivot_table(
            self.df[self.df['mode'] == mode],
            index='matching',
            columns=self.rp.cases[mode],
            values='atkinson_trip_avg_price_std'
        ).T

        for col in data.columns:
            plt.plot(data.index, data[col], 'o--', label=col, c=self.colors[col], alpha=0.8, markersize=4)
            plt.errorbar(data.index, data[col], yerr=data_err[col], fmt='none',
                         c=self.colors[col], label=None, legend=False,
                         alpha=0.8, capsize=3)
        #plt.ylim([0, np.nanmax(data).max() * 1.2])
        plt.legend()
        plt.grid()
        plt.xlabel(self.rp.cases[mode])
        plt.ylabel(r'Atkinson index ($\varepsilon = 0.5$) of incomes')

        plt.savefig('figs/' + self.rp.base + '_entropy_ratio_online_' + mode + '.png', dpi=300)


if __name__=='__main__':
    rv = ResultVisualizer(sys.argv[1])
    rv.create_income_plot('fixed_taxis')
    rv.create_income_plot('fixed_ratio')
    rv.create_requests_completed_plot('fixed_taxis')
    rv.create_requests_completed_plot('fixed_ratio')
    rv.create_entropy_plot('fixed_taxis')
    rv.create_entropy_plot('fixed_ratio')