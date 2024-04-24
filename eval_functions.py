import pandas as pd
import numpy as np
import mizani.formatters as mizani
import plotnine as p9
import tabulate as tab

TIME_MIN = 0.01
TIMEOUT = 120

def read_file(filename):
    """Reads a CSV file into Panda's data frame"""
    df_loc = pd.read_csv(
        filename,
        sep=";",
        )
    return df_loc

def scatter_plot(df, x_tool, y_tool, clamp=True, clamp_domain=[TIME_MIN, TIMEOUT], xname=None, yname=None, log=True, width=6, height=6, show_legend=True):
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

    # formatter for axes' labels
    ax_formatter = mizani.custom_format('{:n}')

    if clamp:  # clamp overflowing values if required
        df = df.copy(deep=True)
        df.loc[df[x_tool] > clamp_domain[1], x_tool] = clamp_domain[1]
        df.loc[df[y_tool] > clamp_domain[1], y_tool] = clamp_domain[1]

    # generate scatter plot
    scatter = p9.ggplot(df)
    scatter += p9.aes(x=x_tool, y=y_tool, color="benchmark")
    scatter += p9.geom_point(size=POINT_SIZE, na_rm=True, show_legend=show_legend, raster=True)
    scatter += p9.labs(x=xname, y=yname)
    scatter += p9.theme(legend_key_width=2)
    scatter += p9.scale_color_hue(l=0.4, s=0.9, h=0.1)

    # rug plots
    scatter += p9.geom_rug(na_rm=True, sides="tr", alpha=0.05, raster=True)

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

    if not show_legend:
        scatter += p9.theme(legend_position='none')

    # generate additional lines
    scatter += p9.geom_abline(intercept=0, slope=1, linetype=DASH_PATTERN)  # diagonal
    scatter += p9.geom_vline(xintercept=clamp_domain[1], linetype=DASH_PATTERN)  # vertical rule
    scatter += p9.geom_hline(yintercept=clamp_domain[1], linetype=DASH_PATTERN)  # horizontal rule

    res = scatter

    return res

def cactus_plot(df, tools, tool_names = None, start = 0, end = None, logarithmic_y_axis=True, width=6, height=6, show_legend=True, legend_loc="upper left"):
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
        legend_loc (str, optional): Legend location. Defaults to "upper left".
    """
    if tool_names == None:
        tool_names = { tool:tool for tool in tools }
    
    if end == None:
        end = len(df)

    concat = dict()

    for tool in tools:
        name = tool_names[tool]
        
        concat[name] = pd.Series(sorted(get_solved(df, tool)[tool + "-runtime"].tolist()))

    # def add_vbs(tools_list, name):
    #     df = df_all_no_nan[[f"{tool}-runtime" for tool in tools_list]].min(axis=1)
    #     concat[name] = pd.Series(sorted(df[df != 120].tolist()))

    # add_vbs(["cvc5-1.1.2", "z3-4.13.0"], "cvc5+Z3")
    # add_vbs(["cvc5-1.1.2", "z3-4.13.0", "z3-noodler-0751e1e-2cddb2f"], "cvc5+Z3+Z3-Noodler")
    # add_vbs(["cvc5-1.1.2", "z3-4.13.0", "ostrich-5dd2e10ca"], "cvc5+z3+ostrich")
    # add_vbs(["cvc5-1.1.2", "z3-4.13.0", "ostrich-5dd2e10ca", "z3-noodler-0751e1e-2cddb2f"], "cvc5+z3+ostrich+noodler")
    # add_vbs([tool for tool in TOOLS if tool != "z3-noodler-0751e1e-2cddb2f"], "vbs without noodler")
    # add_vbs(TOOLS, "vbs")

    concat = pd.DataFrame(concat)


    plt = concat.plot.line(grid=True, fontsize=10, lw=2, figsize=(width, height))
    ticks = np.linspace(start, end, 5, dtype=int)
    plt.set_xticks(ticks)
    plt.set_xlim([start, end])
    plt.set_ylim([0.1, 120])
    if logarithmic_y_axis:
        plt.set_yscale('log')
    plt.set_xlabel("Instances", fontsize=16)
    plt.set_ylabel("Runtime [s]", fontsize=16)

    if show_legend:
        plt.legend(loc=legend_loc)
        # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        # plt.axvline(x=end)

        # plt.get_legend().remove()
        # figlegend = pylab.figure(figsize=(4,4))
        # figlegend.legend(plt.get_children(), concat.columns, loc='center', frameon=False)
        # figlegend.savefig(f"graphs/fig-cactus-{file_name}-legend.pdf", dpi=1000, bbox_inches='tight')
        # plt.figure.savefig(f"graphs/fig-cactus-{file_name}.pdf", dpi=1000, bbox_inches='tight')

    plt.figure.tight_layout()
    # plt.figure.savefig("cactus.pdf", transparent=True)
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
    pt = df
    pt = pt[((pt[tool+"-result"].str.strip() != 'sat') & (pt[tool+"-result"].str.strip() != 'unsat') & (pt[tool+"-result"].str.strip() != 'unknown') & (pt[tool+"-result"].str.strip() != 'TO') & (pt[tool+"-result"].str.strip() != 'ERR'))]
    return pt

def get_solved(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is a result, i.e., either 'sat' or 'unsat'"""
    pt = df
    pt = pt[(pt[tool+"-result"].str.strip() == 'sat')|(pt[tool+"-result"].str.strip() == 'unsat')]
    return pt

def get_unknowns(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is 'unknown'"""
    pt = df
    pt = pt[(pt[tool+"-result"].str.strip() == 'unknown')]
    return pt

def get_timeouts(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is timeout, i.e., 'TO'"""
    pt = df
    pt = pt[(pt[tool+"-result"].str.strip() == 'TO')]
    return pt

def get_errors(df, tool):
    """Returns dataframe containing rows of df, where df[tool-result] is some error or memout, i.e., 'ERR'"""
    pt = df
    pt = pt[((pt[tool+"-result"].str.strip() != 'sat') & (pt[tool+"-result"].str.strip() != 'unsat') & (pt[tool+"-result"].str.strip() != 'unknown') & (pt[tool+"-result"].str.strip() != 'TO'))]
    return pt

def simple_table(df, tools, benches, separately=False):
    """Prints a simple table with statistics for each tools.

    Args:
        df (Dataframe): _description_
        tools (list): List of tools.
        benches (list): List of benchmark sets.
        separately (bool, optional): Should we print table for each benchmark separately. Defaults to False.
    """

    result = ""

    def print_table_from_full_df(df):
        header = ["tool", "✅", "❌", "avg", "med", "time", "sat", "unsat", "unknown", "TO", "MO+ERR", "other"]
        result = ""
        result += f"# of formulae: {len(df)}\n"
        result += "--------------------------------------------------\n"
        table = [header]
        for tool in tools:
            sat = len(df[df[tool + "-result"] == " sat"])
            unsat = len(df[df[tool + "-result"] == " unsat"])
            solved = get_solved(df, tool)[f"{tool}-runtime"]
            avg_solved = solved.mean()
            median_solved = solved.median()
            total_solved = solved.sum()
            unknown = len(get_unknowns(df, tool))
            to = len(get_timeouts(df, tool))
            err = len(get_errors(df, tool))
            other = len(get_invalid(df, tool))
            table.append([tool, sat+unsat, unknown+to+err+other, avg_solved, median_solved, total_solved, sat, unsat, unknown, to, err, other])
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
