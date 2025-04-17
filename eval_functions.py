import pandas as pd
import json
import numpy as np
import mizani.formatters as mizani
import plotnine as p9
import tabulate as tab
from argparse import Namespace
import io
import os
import sys
from enum import Enum
import matplotlib as mpl

import pyco_proc
from pyco_proc import StatsFormat, StatsDestination

def read_latest_result_file(bench, tool, timeout):
    assert tool != ""

    #substring to filter files with the same timeout
    timeout_str = f"to{timeout}-"
    matching_files = []
    for root, _, files in os.walk(bench):
        for file in files:
            if tool in file and timeout_str in file:
                matching_files.append(os.path.join(root, file))
    if not matching_files:
        print(f"WARNING: {tool} has no .tasks file for {bench}")
        return ""
    latest_file_name = sorted(matching_files, key = lambda x: x[-23:])[-1]
    with open(latest_file_name) as latest_file:
        return latest_file.read()


def load_benches(benches, tools, bench_selection, benchmark_to_group, timeout = 120):
    dfs = dict()
    for bench in benches:
        input = ""
        for tool in tools:
            assert tool != ""
            input += read_latest_result_file(bench, tool, timeout)
        has_error_bench = "django" in benches or "biopython" in benches or "thefuck" in benches
        input = pyco_proc.proc_res(io.StringIO(input), Namespace(csv=True,html=False,text=False,tick=False,stats=StatsDestination.OUTPUT_FILE, stats_format=StatsFormat.JSON, ignore_error=has_error_bench))
        df = pd.read_csv(
                io.StringIO(input),
                sep=";",
                dtype='unicode',
        )
        # for the case one tool has only timeouts
        for tool in tools:
            if tool+"-result" not in df.keys():
                df[tool+"-result"] = "TO"
        for key in df.keys():
            if key.endswith("-stats"):
                df[key] = df[key] \
                    .apply(lambda value: value.replace("TO", "{}").replace("ERR", "{}")) \
                    .apply(lambda value: value.replace("###", "").replace("\\", "")) \
                    .apply(json.loads)

                # TODO: Bugfix for the incorrectly named value.
                #  Remove when "str-num-proc-underapprox-solved-preprocess" is no longer being generated and "str-num-proc-length-solved-preprocess" is being correctly generated instead.
                def rename_underapprox_solved_preprocess_to_length_solved_preprocess(stats_dict):
                    if "str-num-proc-underapprox-solved-preprocess" in stats_dict:
                        stats_dict["str-num-proc-length-solved-preprocess"] = stats_dict.pop("str-num-proc-underapprox-solved-preprocess")
                    return stats_dict
                df[key] = df[key].apply(rename_underapprox_solved_preprocess_to_length_solved_preprocess)
        df["benchmark"] = bench
        df["benchmark-group"] = benchmark_to_group[bench]
        dfs[bench] = df

    # tools_no_dates = ['-'.join(tool.split("-")[:-5]) for tool in tools]

    # we select only columns with used tools
    df_stats = pd.concat(dfs, ignore_index=True)[["benchmark", "benchmark-group", "name"] + [f"{tool}-stats" for tool in tools if "stats" in tool]]

    df_runtime_result = pd.concat(dfs, ignore_index=True)[["benchmark", "benchmark-group", "name"] + [f(tool) for tool in tools for f in (lambda x: x + "-result", lambda x: x + "-runtime")]]

    for tool in tools:
        # set runtime to the given parameter for nonsolved instances (unknown, TO, ERR or something else)
        df_runtime_result.loc[(df_runtime_result[f"{tool}-result"] != "sat")&(df_runtime_result[f"{tool}-result"] != "unsat"), f"{tool}-runtime"] = timeout
        # runtime columns should be floats
        df_runtime_result[f"{tool}-runtime"] = df_runtime_result[f"{tool}-runtime"].astype(float)

    if bench_selection == "INT_CONVS":
        # we select only those formulae that contain to_int/from_int
        with open("int_convs_not-full_str_int.txt") as file:
            # fsi_not_conv is a list of formulae from full_str_int that do not contain to_int/from_int
            fsi_not_conv = file.read().splitlines()
        with open("int_convs-str_small_rw.txt") as file:
            # ssr_conv is a list of formulae from str_small_rw that contain to_int/from_int
            ssr_conv = file.read().splitlines()
        with open("int_convs-stringfuzz.txt") as file:
            # sf_conv is a list of formulae from stringfuzz that contain to_int/from_int
            sf_conv = file.read().splitlines()
        df_runtime_result = df_runtime_result[(df_runtime_result.benchmark != "full_str_int")|(~(df_runtime_result.name.isin(fsi_not_conv)))]
        df_runtime_result = df_runtime_result[((df_runtime_result.benchmark != "str_small_rw")&(df_runtime_result.benchmark != "stringfuzz"))|((df_runtime_result.name.isin(ssr_conv))|(df_runtime_result.name.isin(sf_conv)))]

    if bench_selection == "QF_S":
        # for woorpje, QF_S benchmarks are those that are not in 20230329-woorpje-lu/track05/
        df_runtime_result = df_runtime_result[(df_runtime_result.benchmark != "woorpje")|(~(df_runtime_result.name.str.contains("/track05/")))]

    if bench_selection == "QF_SLIA":
        # for woorpje, QF_SLIA benchmarks are those that are in 20230329-woorpje-lu/track05/
        df_runtime_result = df_runtime_result[(df_runtime_result.benchmark != "woorpje")|(df_runtime_result.name.str.contains("/track05/"))]

    df_all = df_runtime_result.merge(df_stats)
    return df_all

def scatter_plot(df, x_tool, y_tool, timeout = 120, clamp=True, clamp_domain=[0.01, 120], xname=None, yname=None, log=True, width=6, height=6, show_legend=True, legend_width=2, file_name_to_save=None, transparent=False, color_by_benchmark=True, color_column="benchmark", value_order=None):
    """Returns scatter plot plotting the values of df[x_tool] and df[y_tool] columns.

    Args:
        df (Dataframe): Dataframe containing the values to plot
        x_tool (str): name of the tool for x-axis
        y_tool (str): name of the tool for x-axis
        clamp (bool, optional): Whether cuts outside of clamp_domain are cut off. Defaults to True.
        clamp_domain (list, optional): The min/max values to plot. Defaults to [TIME_MIN, TIMEOUT].
        xname (str, optional): Name of the x axis. Defaults to None, in that case uses xcol.
        yname (str, optional): Name of the y axis. Defaults to None, in that case uses ycol.
        log (bool, optional): Use logarithmic scale. Defaults to False.
        width (int, optional): Figure width in inches. Defaults to 6.
        height (int, optional): Figure height in inches. Defaults to 6.
        show_legend (bool, optional): Print legend. Defaults to True.
        file_name_to_save (str, optional): If not None, save the result to file_name_to_save.pdf. Defaults to None.
        transparent (bool, optional): Whether the generated plot should have transparent background. Defaults to False.
        color_by_benchmark (bool, optional): Whether the dots should be colored based on the benchmark (if not, there will be just one color). Defaults to True.
        color_column (str, optional): Name of the column to use for coloring. Defaults to 'benchmark'.
    """
    assert len(clamp_domain) == 2
    
    mpl.rcParams['pdf.fonttype'] = 42  # Use Type 1 fonts for PDF output
    mpl.rcParams['ps.fonttype'] = 42  # Use Type 1 fonts for PS output

    POINT_SIZE = 1.0
    DASH_PATTERN = (0, (6, 2))

    if xname is None:
        xname = x_tool
    if yname is None:
        yname = y_tool

    x_tool = x_tool+"-runtime"
    y_tool = y_tool+"-runtime"

    if show_legend:
        width += legend_width

    # formatter for axes' labels
    ax_formatter = mizani.custom_format('{:n}')

    if clamp:  # clamp overflowing values if required
        clamp_domain[1] = timeout
        df = df.copy(deep=True)
        df.loc[df[x_tool] > clamp_domain[1], x_tool] = clamp_domain[1]
        df.loc[df[y_tool] > clamp_domain[1], y_tool] = clamp_domain[1]

    # generate scatter plot
    scatter = p9.ggplot(df)
    if color_by_benchmark:
        scatter += p9.aes(x=x_tool, y=y_tool, color=color_column,)
        scatter += p9.geom_point(size=POINT_SIZE, na_rm=True, show_legend=show_legend, raster=True)
        # rug plots
        scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True)
    else:
        scatter += p9.aes(x=x_tool, y=y_tool, \
        color=color_column, \
        )
        scatter += p9.geom_point(size=POINT_SIZE, na_rm=True, show_legend=show_legend, raster=True, color="orange")
        # rug plots
        scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True, color="orange")
    scatter += p9.labs(x=xname, y=yname)
    scatter += p9.theme(legend_key_width=2)
    scatter += p9.scale_color_hue(l=0.4, s=0.9, h=0.1)

    if log:  # log scale
        scatter += p9.scale_x_log10(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_log10(limits=clamp_domain, labels=ax_formatter)
    else:
        scatter += p9.scale_x_continuous(limits=clamp_domain, labels=ax_formatter)
        scatter += p9.scale_y_continuous(limits=clamp_domain, labels=ax_formatter)

    # scatter += p9.theme_xkcd()
    scatter += p9.theme_bw()
    scatter += p9.theme(panel_grid_major=p9.element_line(color='#666666', alpha=0.5))
    scatter += p9.theme(panel_grid_minor=p9.element_blank())
    scatter += p9.theme(figure_size=(width, height))
    scatter += p9.theme(axis_text=p9.element_text(size=24, color="black", family="Helvetica"))
    scatter += p9.theme(axis_title=p9.element_text(size=24, color="black", family="Helvetica"))
    scatter += p9.theme(legend_text=p9.element_text(size=12))
    if transparent:
        scatter += p9.theme(
            plot_background=p9.element_blank(),
            panel_background = p9.element_rect(alpha=0.0),
            panel_border = p9.element_rect(colour = "black"),
            legend_background=p9.element_rect(alpha=0.0),
            legend_box_background=p9.element_rect(alpha=0.0),
        )

    if not show_legend:
        scatter += p9.theme(legend_position='none')

    # generate additional lines
    scatter += p9.geom_abline(intercept=0, slope=1, linetype=DASH_PATTERN)  # diagonal
    scatter += p9.geom_vline(xintercept=clamp_domain[1], linetype=DASH_PATTERN)  # vertical rule
    scatter += p9.geom_hline(yintercept=clamp_domain[1], linetype=DASH_PATTERN)  # horizontal rule

    if file_name_to_save != None:
        scatter.save(filename=f"{file_name_to_save}.pdf", format="pdf", dpi=500, verbose=False)

    return scatter

def cactus_plot(df, tools, timeout = 120, tool_names = None, start = 0, end = None, logarithmic_y_axis=True, width=6, height=6, show_legend=True, put_legend_outside=False, file_name_to_save=None, num_of_x_ticks=5):
    """Returns cactus plot (sorted runtimes of each tool in tools). To print the result use result.figure.savefig("name_of_file.pdf", transparent=True).

    Args:
        df (Dataframe): Dataframe containing for each tool in tools column tool-result and tool-runtime containing the result and runtime for each benchmark.
        tools (list): List of tools to plot.
        tool_names (dict, optional): Maps each tool to its name that is used in the legend. If not set (=None), the names are taken directly from tools.
        start (int, optional): The starting position of the x-axis. Defaults to 0.
        end (int, optional): The ending position of the x-axis. If not set (=None), defaults to number of benchmarks, i.e. len(df).
        logarithmic_y_axis (bool, optional): Use logarithmic scale for the y-axis. Defaults to True.
        width (int, optional): Figure width in inches. Defaults to 6.
        height (int, optional): Figure height in inches. Defaults to 6.
        show_legend (bool, optional): Print legend. Defaults to True.
        put_legend_outside (bool, optional): Whether to put legend outside the plot. Defaults to False.
        file_name_to_save (str, optional): If not None, save the result to file_name_to_save.pdf. Defaults to None.
        num_of_x_ticks (int, optional): Number of ticks on the x-axis. Defaults to 5.
    """
    if tool_names == None:
        tool_names = { tool:tool for tool in tools }

    if end == None:
        end = len(df)

    concat = dict()

    for tool in tools:
        name = tool_names[tool]

        concat[name] = pd.Series(sorted(get_solved(df, tool)[tool + "-runtime"].tolist()))

    concat = pd.DataFrame(concat)


    plt = concat.plot.line(figsize=(width, height))
    ticks = np.linspace(start, 150200, num_of_x_ticks, dtype=int)
    ticks = ticks[:-1]
    ticks = np.append(ticks, 150287)
    plt.set_xticks(ticks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.set_xlim([start, end])
    plt.set_ylim([0.1, timeout])
    if logarithmic_y_axis:
        plt.set_yscale('log')
    plt.set_xlabel("Instances", fontsize=16)
    plt.set_ylabel("Runtime [s]", fontsize=16)

    if show_legend:
        if put_legend_outside:
            plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left',framealpha=0.1)
        else:
            plt.legend(loc='upper left',framealpha=0.1)

        # plt.axvline(x=end)

        # plt.get_legend().remove()
        # figlegend = pylab.figure(figsize=(4,4))
        # figlegend.legend(plt.get_children(), concat.columns, loc='center', frameon=False)
        # figlegend.savefig(f"graphs/fig-cactus-{file_name}-legend.pdf", dpi=1000, bbox_inches='tight')
        # plt.figure.savefig(f"graphs/fig-cactus-{file_name}.pdf", dpi=1000, bbox_inches='tight')

    plt.figure.tight_layout()
    if file_name_to_save != None:
        plt.figure.savefig(f"{file_name_to_save}.pdf", transparent=True)
    return plt

def sanity_check(df, tool, compare_with):
    """Returns dataframe containing rows of df, where df[tool-result] is different (sat vs. unsat) than the result of any of the tools in compare_with

    Args:
        compare_with (list): List of tools to compare with.
    """
    all_bad = []
    for tool_other in compare_with:
        pt = df
        pt = pt[((pt[tool+"-result"].str.strip() == 'sat') & (pt[tool_other+"-result"].str.strip() == 'unsat') | (pt[tool+"-result"].str.strip() == 'unsat') & (pt[tool_other+"-result"].str.strip() == 'sat'))]
        all_bad.append(pt)
    return pd.concat(all_bad).drop_duplicates()

def get_invalid(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is not a valid result (sat/unsat/unknown/TO/ERR)"""
    return df[((df[tool+"-result"].str.strip() != 'sat') & (df[tool+"-result"].str.strip() != 'unsat') & (df[tool+"-result"].str.strip() != 'unknown') & (df[tool+"-result"].str.strip() != 'TO') & (df[tool+"-result"].str.strip() != 'ERR'))]

def get_solved(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is a result, i.e., either 'sat' or 'unsat'"""
    return df[(df[tool+"-result"].str.strip() == 'sat')|(df[tool+"-result"].str.strip() == 'unsat')]

def get_unknowns(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is 'unknown'"""
    return df[(df[tool+"-result"].str.strip() == 'unknown')]

def get_timeouts(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is timeout, i.e., 'TO'"""
    return df[(df[tool+"-result"].str.strip() == 'TO')]

def get_errors(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is some error or memout, i.e., 'ERR'"""
    return df[(df[tool+"-result"].str.strip() == 'ERR')]

def get_sat(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is 'sat'"""
    return df[(df[tool+"-result"].str.strip() == 'sat')]

def get_unsat(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is 'unsat'"""
    return df[(df[tool+"-result"].str.strip() == 'unsat')]

def simple_table(df, tools, benches, separately=False, times_from_solved=True):
    """Prints a simple table with statistics for each tools.

    Args:
        df (Dataframe): data
        tools (list): List of tools.
        benches (list): List of benchmark sets.
        separately (bool, optional): Should we print table for each benchmark separately. Defaults to False.
        times_from_solved (bool, optional): Should we compute total, average and median time from only solved instances. Defaults to True.
    """

    result = ""

    def print_table_from_full_df(df):
        header = ["tool", "✅", "❌", "time", "avg", "med", "std", "sat", "unsat", "unknown", "TO", "MO+ERR", "other"]
        result = ""
        result += f"# of formulae: {len(df)}\n"
        result += "--------------------------------------------------\n"
        table = [header]
        for tool in tools:
            sat = len(df[df[tool + "-result"].str.strip() == "sat"])
            unsat = len(df[df[tool + "-result"].str.strip() == "unsat"])
            runtime_col = df[f"{tool}-runtime"]
            if times_from_solved:
                runtime_col = get_solved(df, tool)[f"{tool}-runtime"]
            avg = runtime_col.mean()
            median = runtime_col.median()
            total = runtime_col.sum()
            std = runtime_col.std()
            unknown = len(get_unknowns(df, tool))
            to = len(get_timeouts(df, tool))
            err = len(get_errors(df, tool))
            other = len(get_invalid(df, tool))
            table.append([tool, sat+unsat, unknown+to+err+other, total, avg, median, std, sat, unsat, unknown, to, err, other])
        result += tab.tabulate(table, headers='firstrow', floatfmt=".2f") + "\n"
        result += "--------------------------------------------------\n\n"
        return result

    if (separately):
        for bench in benches:
            result += f"Benchmark {bench}\n"
            result += print_table_from_full_df(df[df["benchmark"] == bench])
    else:
        result += print_table_from_full_df(df[df["benchmark"].isin(benches)])

    return result

def add_vbs(df, tools_list, name = None):
    """Adds virtual best solvers from tools in tool_list

    Args:
        df (Dataframe): data
        tools_list (list): list of tools
        name (str, optional): Name of the vbs used for the new columns. If not set (=None), the name is generated from the name of tools in tool_list.

    Returns:
        Dataframe: same as df but with new columns for the vbs
    """
    if name == None:
        name = "+".join(tools_list)
    df[f"{name}-runtime"] = df[[f"{tool}-runtime" for tool in tools_list]].min(axis=1)
    def get_result(row):
        nonlocal tools_list
        if "sat" in [str(row[f"{tool}-result"]).strip() for tool in tools_list]:
            return "sat"
        elif "unsat" in [str(row[f"{tool}-result"]).strip() for tool in tools_list]:
            return "unsat"
        else:
            return "unknown"
    df[f"{name}-result"] = df.apply(get_result, axis=1) # https://stackoverflow.com/questions/26886653/create-new-column-based-on-values-from-other-columns-apply-a-function-of-multi
    return df

def fuck():
    print("fuck")


def q75(x):
    return np.percentile(a=x, q=75)


def q25(x):
    return np.percentile(a=x, q=25)


def get_stats_total(df: pd.DataFrame, tool: str, benchmarks: list[str]) -> pd.DataFrame:
    return df.loc[df["benchmark"].isin(benchmarks)].drop(["name", "benchmark", f"{tool}-result"], axis=1).agg(['mean', 'std', 'min', 'max', 'sum', 'var', 'median'])


class StatsDataType(Enum):
  INT = 0
  FLOAT = 1
  STRING = 2


STATS_DATA_TYPES = {
    "added-eqs": StatsDataType.INT,
    "arith-bound-propagations-lp": StatsDataType.INT,
    "arith-eq-adapter": StatsDataType.INT,
    "arith-fixed-eqs": StatsDataType.INT,
    "arith-lower": StatsDataType.INT,
    "arith-make-feasible": StatsDataType.INT,
    "arith-max-columns": StatsDataType.INT,
    "arith-max-rows": StatsDataType.INT,
    "arith-upper": StatsDataType.INT,
    "binary-propagations": StatsDataType.INT,
    "decisions": StatsDataType.INT,
    "del-clause": StatsDataType.INT,
    "final-checks": StatsDataType.INT,
    "noodler-final_checks": StatsDataType.INT,
    "max-memory": StatsDataType.FLOAT,
    "memory": StatsDataType.FLOAT,
    "mk-bool-var": StatsDataType.INT,
    "mk-clause": StatsDataType.INT,
    "mk-clause-binary": StatsDataType.INT,
    "num-allocs": StatsDataType.INT,
    "num-checks": StatsDataType.INT,
    "propagations": StatsDataType.INT,
    "rlimit-count": StatsDataType.INT,
    "str-num-proc-stabilization-finish": StatsDataType.INT,
    "str-num-proc-stabilization-start": StatsDataType.INT,
    "time": StatsDataType.FLOAT,
    "total-time": StatsDataType.FLOAT,
    "conflicts": StatsDataType.INT,
    "str-num-solved-preprocess": StatsDataType.INT,
    "str-num-proc-multi-memb-heur-finish": StatsDataType.INT,
    "str-num-proc-multi-memb-heur-start": StatsDataType.INT,
    "arith-diseq": StatsDataType.INT,
    "arith-offset-eqs": StatsDataType.INT,
    "str-num-proc-underapprox-finish": StatsDataType.INT,
    "str-num-proc-underapprox-start": StatsDataType.INT,
    "arith-gcd-calls": StatsDataType.INT,
    "arith-patches": StatsDataType.INT,
    "arith-patches-success": StatsDataType.INT,
    "str-num-proc-length-finish": StatsDataType.INT,
    "str-num-proc-length-start": StatsDataType.INT,
    "str-num-proc-single-memb-heur-finish": StatsDataType.INT,
    "str-num-proc-single-memb-heur-start": StatsDataType.INT,
    "solve-eqs-elim-vars": StatsDataType.INT,
    "solve-eqs-steps": StatsDataType.INT,
    "str-num-proc-nielsen-finish": StatsDataType.INT,
    "str-num-proc-nielsen-start": StatsDataType.INT,
    "str-num-proc-unary-finish": StatsDataType.INT,
    "str-num-proc-unary-start": StatsDataType.INT,
    "str-num-proc-stabilization-solved-preprocess": StatsDataType.INT,
    "str-num-proc-length-solved-preprocess": StatsDataType.INT,
}


def get_stats_dfs(df, tool, order=None):
    global STATS_DATA_TYPES
    TOOL_STATS_STR = f"{tool}-stats"
    TOOL_RUNTIME_STR = f"{tool}-runtime"
    TOOL_RESULT_STR = f"{tool}-result"
    additional_columns_strs = ['name', "benchmark", TOOL_RUNTIME_STR, TOOL_RESULT_STR]
    df_stats = pd.json_normalize(df[TOOL_STATS_STR].copy())

    stats_columns_to_keep_strs = []
    for column in df_stats.keys():
        if column.endswith("-start") or \
        column.endswith("-finish") or \
        column.endswith("-preprocess") or \
        column in ["noodler-final_checks"]:
            stats_columns_to_keep_strs.append(column)

    df_stats = df_stats[stats_columns_to_keep_strs]
    df_stats = pd.concat([df[additional_columns_strs], df_stats], axis=1)
    df_stats.replace({"benchmark": {"snia": "kaluza"}}, inplace=True)

    for column in df_stats:
        if column not in additional_columns_strs:
            if STATS_DATA_TYPES[column] == StatsDataType.INT:
                df_stats[column] = df_stats[column].astype('Int64')
            elif STATS_DATA_TYPES[column] == StatsDataType.FLOAT:
                df_stats[column] = df_stats[column].astype(float)
            elif STATS_DATA_TYPES[column] == StatsDataType.STRING:
                df_stats[column] = df_stats[column].astype(str)

    # for column in df_stats.columns:
    #     if column.endswith("-start") or column.endswith("-finish"):
    #         column_per_final_check_name = f"{column}-per-final-check"
    #         df_stats[column_per_final_check_name] = df_stats[column] / df_stats["final-checks"]

    #         df_stats[column_per_final_check_name] = df_stats[column_per_final_check_name].astype('Int64')

    if order:
        if "unknown" not in order:
            order.append("unknown")
        df_stats["benchmark"] = pd.Categorical(df_stats["benchmark"], order)
        df_stats.sort_values("benchmark")

    fill_nan_dict = {}
    for column in df_stats.keys():
        if column in ["name", "benchmark"]:
            continue
        fill_nan_dict[column] = 0
    # fill_nan_dict["name"] = "unknown"
    # fill_nan_dict["benchmark"] = "unknown"
    df_stats_zeroed_nans = df_stats.fillna(value=fill_nan_dict, inplace=False)

    return df_stats, df_stats_zeroed_nans


def get_stats_grouped_by_benchmark(df, tool):
    df_stats_grouped_by_benchmark = df.drop(["name", f"{tool}-result"], axis=1).groupby(["benchmark"], observed=True).agg(['sum', ])

    return df_stats_grouped_by_benchmark


def get_stats_grouped_by_benchmark_counts(df, tool):
    return df.drop(["name", f"{tool}-result"], axis=1).groupby(["benchmark"], observed=True).agg(['count'])


def get_stats_characteristics_grouped_by_benchmark_characteristics(df, tool):
    return df.drop(["name", f"{tool}-result"], axis=1).groupby(["benchmark"]).agg(['mean', 'std', 'min', 'max', 'sum', 'var', q25, 'median', q75])


def group_to_benchmark_groups(df, benchmark_to_group, order=None):
    df_grouped = df.copy()
    # df_grouped.replace({"benchmark": benchmark_to_group}, inplace=True)
    df_grouped["benchmark"] = df_grouped["benchmark"] \
        .map(benchmark_to_group) \
        # .astype('category')
    # df_grouped["benchmark"].cat.set_categories(order)
    # df_grouped["benchmark"].sort_values(inplace=True)
    if order:
        if "unknown" not in order:
            order.append("unknown")
        df_grouped["benchmark"] = pd.Categorical(df_grouped["benchmark"], order)
        df_grouped.sort_values("benchmark")
    return df_grouped


def get_stats_paper(df):
    df_paper = pd.DataFrame()
    # df_paper["name"] = df["name"]
    df_paper["benchmark"] = df["benchmark"]
    df_paper["noodler-final_checks"] = df["noodler-final_checks"]

    df_paper["str-num-proc-memb-heur-start"] = \
        df["str-num-proc-multi-memb-heur-start"] \
        + df["str-num-proc-single-memb-heur-finish"]
        # + df["str-num-proc-single-memb-heur-start"] \
    df_paper["str-num-proc-memb-heur-finish"] = df["str-num-proc-multi-memb-heur-finish"] + df["str-num-proc-single-memb-heur-finish"]

    df_paper["str-num-proc-nielsen-start"] = df["str-num-proc-nielsen-start"]
    df_paper["str-num-proc-nielsen-finish"] = df["str-num-proc-nielsen-finish"]

    df_paper["str-num-proc-length-start"] = df["str-num-proc-length-start"] + df["str-num-proc-length-solved-preprocess"]
    df_paper["str-num-proc-length-finish"] = df["str-num-proc-length-finish"] + df["str-num-proc-length-solved-preprocess"]
    df_paper["str-num-proc-length-solved-preprocess"] = df["str-num-proc-length-solved-preprocess"]


    df_paper["str-num-proc-stabilization-start"] = \
        df["str-num-proc-stabilization-start"] \
        + df["str-num-proc-underapprox-finish"] \
        + df["str-num-proc-stabilization-solved-preprocess"]
        # df["str-num-proc-underapprox-start"] \
    df_paper["str-num-proc-stabilization-finish"] = df["str-num-proc-stabilization-finish"] + df["str-num-proc-underapprox-finish"] \
        + df["str-num-proc-stabilization-solved-preprocess"]
    df_paper["str-num-proc-stabilization-solved-preprocess"] = df["str-num-proc-stabilization-solved-preprocess"]

    return df_paper

def get_stats_per_benchmark_paper(df):
    df_paper = get_stats_paper(df)
    df_stats_per_benchmark_sum = df_paper.groupby("benchmark", observed=True).sum()

    for key in df_stats_per_benchmark_sum.keys():
        if key.endswith("-finish") or key.endswith("-start") or key.endswith("-solved-preprocess"):
            df_stats_per_benchmark_sum[key] = df_stats_per_benchmark_sum[key] / df_stats_per_benchmark_sum["noodler-final_checks"] * 100
            df_stats_per_benchmark_sum[key] = df_stats_per_benchmark_sum[key].round(decimals=2)

    return df_stats_per_benchmark_sum


def write_latex_table_body(df, float_format="{:.2f}", format_index_name=True, index_to_latex=None):
    def format_index_name_default(name):
        if index_to_latex and name in index_to_latex:
            return index_to_latex[name]

        return name

    df_table = df
    if format_index_name:
        df_table = df.rename(index=format_index_name_default)
    return df_table.to_latex(buf=None, columns=None, header=False, index=True, na_rep='NaN', formatters=None, float_format=float_format.format, sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None, label=None, position=None).splitlines()


def table_solved_time(df, df_all, benchmarks, benchmark_to_latex, tool_to_latex, per_column="benchmark"):
    table_lines = write_latex_table_body(df, format_index_name=True, index_to_latex=tool_to_latex, float_format="{:,.0f}")
    table_lines.insert(2, "")
    table_lines.insert(3, "")
    table_lines.insert(4, "")
    table_lines.insert(5, "")
    i = 2
    for benchmark in benchmarks:
        table_lines[2] += "& " + f"{benchmark_to_latex[benchmark]} & "
        if benchmark == "all":
            count = df_all.count().iloc[0]
        else:
            count = df_all[df_all[per_column] == benchmark].count().iloc[0]
        table_lines[3] += "& " + f"({count:,}) & "
        table_lines[4] += "\\cmidrule[lr]{" + str(i) + "-" + str(i + 1) + "}"
        i += 2
        table_lines[5] += f"& solved & time "
    table_lines[2] += "\\\\"
    table_lines[3] += "\\\\"
    table_lines[5] += "\\\\"
    del table_lines[6]
    table_lines = table_lines[1:-1]
    return table_lines


def solved_time_transpose_per_benchmark(df_solved_time):
    df_solved_time_transposed = df_solved_time.transpose()
    concat_rows = []
    for index, _ in df_solved_time_transposed.iterrows():
        if not index.endswith("-result"):
            continue

        index_result_name = index
        index_runtime_name = index_result_name.replace("-result", "-runtime")
        procedure_name = index_result_name.replace("-result", "")
        result_row = df_solved_time_transposed.loc[[index_result_name]]
        runtime_row = df_solved_time_transposed.loc[[index_runtime_name]]

        concat_row = [procedure_name]
        values_result = list(result_row.values)
        values_runtime = list(runtime_row.values)
        for val_result, val_runtime in zip(values_result, values_runtime):
            for val_result, val_runtime in zip(val_result, val_runtime):
                concat_row += [val_result, val_runtime]

        concat_rows.append(concat_row)

    columns = ["tool"]
    for column in df_solved_time_transposed.keys():
        columns.append(f"{column}-result")
        columns.append(f"{column}-runtime")
    df_concat_rows = pd.DataFrame(concat_rows, columns=columns)
    df_concat_rows.set_index("tool", inplace=True)
    for column in df_concat_rows.columns:
        if column.endswith("-result"):
            df_concat_rows[column] = df_concat_rows[column].apply(lambda x: '{:,}'.format(x))
    
    return df_concat_rows
