import pandas as pd
import numpy as np
import mizani.formatters as mizani
import plotnine as p9
import tabulate as tab
import pyco_proc
from argparse import Namespace
import io
import os

def read_latest_result_file(bench, tool):
    assert tool != ""

    matching_files = []
    for root, _, files in os.walk(bench):
        for file in files:
            if tool in file:
                matching_files.append(os.path.join(root, file))
    if not matching_files:
        print(f"WARNING: {tool} has no .tasks file for {bench}")
        return ""
    latest_file_name = sorted(matching_files, key = lambda x: x[-23:])[-1]
    with open(latest_file_name) as latest_file:
        return latest_file.read()


def load_benches(benches, tools, bench_selection):
    dfs = dict()
    for bench in benches:
        input = ""
        for tool in tools:
            assert tool != ""
            input += read_latest_result_file(bench, tool)
        input = pyco_proc.proc_res(io.StringIO(input), Namespace(csv=True,html=False,text=False,tick=False,stats=None))
        df = pd.read_csv(
                io.StringIO(input),
                sep=";",
                dtype='unicode',
        )
        df["benchmark"] = bench
        dfs[bench] = df

    # we select only columns with used tools
    df_all = pd.concat(dfs, ignore_index=True)[["benchmark"] + ["name"] + [f(tool) for tool in tools for f in (lambda x: x+"-result", lambda x: x+"-runtime")]]

    for tool in tools:
        # set runtime to 120 for nonsolved instances (unknown, TO, ERR or something else)
        df_all.loc[(df_all[f"{tool}-result"] != "sat")&(df_all[f"{tool}-result"] != "unsat"), f"{tool}-runtime"] = 120
        # runtime columns should be floats
        df_all[f"{tool}-runtime"] = df_all[f"{tool}-runtime"].astype(float)

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
        df_all = df_all[(df_all.benchmark != "full_str_int")|(~(df_all.name.isin(fsi_not_conv)))]
        df_all = df_all[((df_all.benchmark != "str_small_rw")&(df_all.benchmark != "stringfuzz"))|((df_all.name.isin(ssr_conv))|(df_all.name.isin(sf_conv)))]
    
    if bench_selection == "QF_S":
        # for woorpje, QF_S benchmarks are those that are not in 20230329-woorpje-lu/track05/
        df_all = df_all[(df_all.benchmark != "woorpje")|(~(df_all.name.str.contains("/track05/")))]

    if bench_selection == "QF_SLIA":
        # for woorpje, QF_SLIA benchmarks are those that are in 20230329-woorpje-lu/track05/
        df_all = df_all[(df_all.benchmark != "woorpje")|(df_all.name.str.contains("/track05/"))]
    
    return df_all

def scatter_plot(df, x_tool, y_tool, clamp=True, clamp_domain=[0.01, 120], xname=None, yname=None, log=True, width=6, height=6, show_legend=True, legend_width=2, file_name_to_save=None, transparent=False, color_by_benchmark=True):
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
    """
    assert len(clamp_domain) == 2

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
        df = df.copy(deep=True)
        df.loc[df[x_tool] > clamp_domain[1], x_tool] = clamp_domain[1]
        df.loc[df[y_tool] > clamp_domain[1], y_tool] = clamp_domain[1]

    # generate scatter plot
    scatter = p9.ggplot(df)
    if color_by_benchmark:
        scatter += p9.aes(x=x_tool, y=y_tool, color="benchmark")
        scatter += p9.geom_point(size=POINT_SIZE, na_rm=True, show_legend=show_legend, raster=True)
        # rug plots
        scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True)
    else:
        scatter += p9.aes(x=x_tool, y=y_tool)
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
    scatter += p9.theme(axis_text=p9.element_text(size=24, color="black"))
    scatter += p9.theme(axis_title=p9.element_text(size=24, color="black"))
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
        scatter.save(filename=f"{file_name_to_save}.pdf", dpi=500, verbose=False)

    return scatter

def cactus_plot(df, tools, tool_names = None, start = 0, end = None, logarithmic_y_axis=True, width=6, height=6, show_legend=True, put_legend_outside=False, file_name_to_save=None, num_of_x_ticks=5):
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
    ticks = np.linspace(start, end, num_of_x_ticks, dtype=int)
    plt.set_xticks(ticks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.set_xlim([start, end])
    plt.set_ylim([0.1, 120])
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
