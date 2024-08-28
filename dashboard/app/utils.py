import os

import streamlit as st  # 1.0.0 o impazzisce
import pandas as pd  # 1.2.4 o impazzisce
import numpy as np

import altair as alt
import plotly.express as px

import copy
import csv
import sys
import time

import functools
from concurrent.futures import ThreadPoolExecutor

workers = 8

MAIN_PATH = r'P:\OneDrive\Dropbox\My_Papers\Glob_Lime'
PYMOB_PATH = os.path.join(MAIN_PATH, 'pymob')
GLEAMS_PATH = os.path.join(MAIN_PATH, 'gleams')
sys.path.append(MAIN_PATH)
sys.path.append(PYMOB_PATH)
sys.path.append(GLEAMS_PATH)
# sys.path.append(r'..\\..\\helpers')
# sys.path.append(r'..\\..\\pymob\\mob')
# sys.path.append(r'../../gleams')

from helpers import sobol as sbl
import mob_utils
from mob import MOB
import gleams_utils


# TODO change _variables (e.g _mob to mob) and understand why hashing do not work


@st.experimental_memo
def check_csv(file):
    """Checks whether the file is a valid well-formed csv """
    csv_file = open(file, 'rb')
    dialect = csv.Sniffer().sniff(csv_file.read(1024))
    # Perform various checks on the dialect (lineseparator,delimiter, etc) to make sure it's sane

    # Reset the read position back to the start of
    # the file before reading any entries.
    csv_file.seek(0)


###mock model
@st.experimental_memo  # (allow_output_mutation=True)
def get_model(X, y, minsplit=5, stopping_value=0.95):
    mob = MOB(minsplit=minsplit, stopping_value=stopping_value)
    mob.fit(X, y)

    return mob


###data
@st.experimental_memo(ttl=60 * 60)
def obtain_data(d=2, m=7, wide=2, bias=0.5, f=sbl.xsinx):  # sourcery skip: inline-immediately-returned-variable
    np.random.seed(42)

    # names and dims
    # names = ["x" + str(i) for i in range(d)]
    names = [str(i) for i in range(d)]
    names.append("y")

    # actual df
    points = (sbl.get_sobol_x(d, m=m) - bias) * wide
    y = np.array([f(points[i, :]) for i in range(2 ** m)]).reshape(-1, 1)

    data = pd.DataFrame(np.c_[points, y], columns=names)

    return data


@st.experimental_memo(ttl=60 * 60)
def obtain_variables(d, c):
    """

    Args:
        d: pandas DataFrame (it is the dataset that we loaded as dataframe from the csv)
        c: column name of the Y variable

    Returns: list of variable names, which are the X variables (take all the columns apart from the Y variable)

    """
    vars = [x for x in d.columns if x != c]

    return vars


# subsetting the original dataset for the current selected variable
# @st.experimental_memo(ttl=60 * 60)
@st.cache(ttl=60 * 60)
def get_original_points(uploaded_dataframe, _nodes):
    """
    Retrieve all datapoints from the uploaded_dataframe, contained in the list of leaves called _nodes

    Args:
        uploaded_dataframe:
        _nodes:
        var:

    Returns:

    """

    init_time = time.time()
    points_ids = []
    # read the uploaded_df row by row, select the ids of the rows contained in the list of nodes,
    # subset the df to obtain only the points contained in the list of nodes
    for id_row, row in uploaded_dataframe.iterrows():
        row_arr = np.array(row)
        if any(node.domain.domain_contains(row_arr, None, False) for node in _nodes):
            points_ids.append(id_row)
    end_time = time.time()
    return uploaded_dataframe.iloc[points_ids], end_time - init_time


# not using it right now, because of the clearing cache bug of st.experimental_memo
def reset_memo_nodes():
    """Reset the list of nodes for the 1d what-if analysis
    and the dataframes with the corresponding points"""
    get_selected_nodes.clear()
    get_synthetic_points.clear()
    get_original_points.clear()


# @st.experimental_memo(ttl=60 * 60)
@st.cache
def get_synthetic_points(_nodes, variable_names):
    """
    Retrieve all Sobol generated datapoints contained in the list of leaves called _nodes

    Parameters
    ----------
    _nodes: list of nodes
    variable_names: list of variable names, ordered as in the sobol points used in the gleams training
                    (hence as in the dataframe passed to gleams in the fit method).
                    Shall contain all the X names and the target variable name at the end

    Returns
    -------
    Dataframe containing the selected points
    """

    init_time = time.time()

    X, y = _nodes[0].leaf_points
    for node in _nodes[1:]:
        X1, y1 = node.leaf_points
        X = np.vstack((X, X1))
        y = np.hstack((y, y1))
    y = y[:, np.newaxis]
    synth_df = pd.DataFrame(np.hstack((X, y)), columns=variable_names)
    end_time = time.time()

    return synth_df, end_time - init_time


### functions for regression
@st.experimental_memo(ttl=60 * 60)
def get_coefficients(_mob, x):
    """
    From a datapoint and a mob fitted model,
    predict the datapoint to find the leaf,
    then extract the coefficients from the leaf.
    For now the intercept is not returned
    Args:
        _mob:
        x:

    Returns:

    """
    node = mob_utils.get_pred_node(_mob.tree, x.squeeze(axis=0))
    coef = node.regression.coef_
    # intercept = node.intercept_

    return coef


@st.cache(ttl=60 * 60, allow_output_mutation=True)
def highlight_cell_in_df(dataframe, value_to_highlight, selected_var, color="orange"):
    """

    Parameters
    ----------
    dataframe: df to apply style changes (background color)
    value_to_highlight: value to look for in the df
    selected_var: variable in which the value should be found
    color: chosen color for the background

    Returns
    -------
    a dataframe with the requested style applied, i.e. the background color
    """

    # this lambda scroll the subset_dataframe looking for the value_to_highlight and applies on it the background color
    func = lambda dataframe: ["background-color: {}".format(color) if val == value_to_highlight else "" for val in
                              dataframe]
    return dataframe.style.apply(func, subset=selected_var).format(precision=2)


@st.cache(ttl=60 * 60, allow_output_mutation=True)
def highlight_row_in_df(dataframe, selected_row_id, color="orange"):
    """

    Parameters
    ----------
    dataframe: df to apply style changes (background color)
    selected_row_id: index of the row to be highlighted
    color: chosen color for the background

    Returns
    -------
    a dataframe with the requested style applied, i.e. the background color
    """

    return dataframe.style.set_properties(subset=pd.IndexSlice[selected_row_id, :],
                                          **{'background-color': str(color)}).format(precision=2)


### functions for plotting
@st.experimental_memo(ttl=60 * 60)
def make_altair_scatterplot(d, columns, tooltip):
    """
    Create altair Scatterplot, of the d Dataframe points


    Args:
        d: DataFrame of points to be plotted in the altair plot
        columns: dictionary, like {"x": variable_name, "y": variable_name} containing
                 Variable names for the altair axes
        tooltip: Labels to be shown when tooltip on the datapoints

    Returns: A Scatterplot of the DataFrame d
    :param _plot_scaling:

    """

    # this is a condition to understand when a point is selected (will be used for displaying the tooltip)
    selected = alt.selection_single(on="mouseover", empty="none")

    # Examples used to create this plot
    # https://altair-viz.github.io/gallery/scatter_tooltips.html
    # https://github.com/domoritz/streamlit-vega-lite-demo/blob/main/demo.py

    x = columns["x"]  # x
    y = columns["y"]  # y
    scatter = alt.Chart(d).mark_circle(size=80).encode(
        alt.X(x),
        alt.Y(y),
        # change color to red when pass over a datapoint for tooltip
        color=alt.condition(selected, alt.value("red"), alt.value("steelblue")),
        tooltip=tooltip
    ).add_selection(selected)

    return scatter


@st.experimental_memo(ttl=60 * 60)
def make_altair_point(x, y, point, tooltip, coef, color="orange"):
    """
    Create a dataframe which is only one row (with values x,y).
    Create the scatterplot of the single point.
    USed to plot the what-if datapoint in the Scatterplot

    Args:
        x:
        y:
        point: dictionary like {"x": name_var, "x_val": var_value} same for y
        tooltip:
        coef: scalar value, of the x variable coefficient

    Returns:

    Parameters
    ----------
    color
    :param _plot_scaling:
    :param axes_names:

    """

    # dataframe with a single point
    p_df = pd.DataFrame()
    p_df[point["x"]] = np.array([point["x_val"]])
    p_df[point["y"]] = np.array([point["y_val"]])

    # add the coef column to the one-line dataframe
    p_df["coef"] = np.array([coef])

    # create a list of things to be displayed in the tooltip (of the what-if datapoint).
    # Add the coeff to the tooltip list
    tips = [x for x in tooltip]
    tips.append("coef")

    # create the Scatterplot of the single point dataframe
    single_point = alt.Chart(p_df).mark_circle(size=150).encode(
        alt.X(x),
        alt.Y(y),
        color=alt.value(color),
        tooltip=tips
    )

    return single_point


@st.experimental_memo(ttl=60 * 60)
def make_altair_plot(d, p, columns, tooltip, coef="N.D."):
    """
    Create altair Scatterplot, with the what-if datapoint with a different color

    Args:
        d:
        p:
        columns:
        tooltip:
        coef:

    Returns:
    :param _plot_scaling:

    """

    # deepcopy is useful to not impact the cache
    # (in this way, we can change only the copy of the plot, keeping the original one fixed)

    # make scatterplot with all the data
    scatterplot = copy.deepcopy(make_altair_scatterplot(d, columns, tooltip))

    x = columns["x"]  # x
    y = columns["y"]  # y

    # scatterplot of the what-if datapoint
    single_point = make_altair_point(x, y, p, tooltip, coef)

    # assemble the scatterplot and the what-if datapoint in a single fig
    fig = alt.layer(scatterplot, single_point).interactive()
    # .configure(autosize="pad")   # it was added to the fig, but I removed it to allow the same scaling of the two pictures
    # explicit delete of the deepcopy
    del scatterplot

    return fig


@st.cache(ttl=60 * 60)
def make_altair_stacked_chart(_nodes, selected_var_name, var_ids_dict):
    """
    This plot shows the importance of the selected_variable when changing from one leaf to another
    (on each leaf domain, there is a bar with value equal to the difference in mob prediction
    of changing the variable from the left bound to the right bound)

    Spiegazione della funzione dal punto di vista del codice:

    prendo la lista dei nodi che mi interessano

    per ogni nodo
        calcolo quanto vale il contributo di ogni feature sugli estremi del nodo (a, b)
            singolo contributo_xi = coefficiente xi * ai

            tupla = (valore_variabile di scorrimento, feature, contributo feature)
            aggiungo la tupla alla lista

    Parameters
    ----------
    selected_var_id
    """

    selected_var_id = list(var_ids_dict.keys())[list(var_ids_dict.values()).index(selected_var_name)]

    # names to be used in the contributions dataframe
    columns = [selected_var_name, "features", "contributes"]

    l = []
    for node in _nodes:
        # save coefficients, intercept and domain of each leaf
        coefficients = node.regression.coef_
        intercept = node.regression.intercept_
        domain = node.domain

        left_bound_sel_var = domain[selected_var_id][0]
        right_bound_sel_var = domain[selected_var_id][1]

        for var_id, v in enumerate(coefficients):
            # TODO: devo considerare che ogni variabile verrà passata come nome della variabile e non come id_var.
            #  Quindi devo ricavarmi l'indice della variabile dal nome, invece di usare l'enumerate
            # Calculate the Contributions of each variable in the leaf.
            # Contribution = increase in mob_prediction, changing the variable from the left bound to the right bound

            # interval bounds
            left_bound_var = domain[var_id][0]
            right_bound_var = domain[var_id][1]

            # single contribution for var
            contr_left = coefficients[var_id] * left_bound_var
            contr_right = coefficients[var_id] * right_bound_var

            c_a_list = [left_bound_sel_var, var_ids_dict[var_id], contr_left]
            c_b_list = [right_bound_sel_var, var_ids_dict[var_id], contr_right]

            # l is a list of lists, containing the contributions left-right for each variable
            l.append(c_a_list)
            l.append(c_b_list)

        # add to l, also the contributions left-right of the intercept
        int_a_list = [left_bound_sel_var, "intercept", intercept]
        int_b_list = [right_bound_sel_var, "intercept", intercept]
        l.append(int_a_list)
        l.append(int_b_list)

    # PROVA PER FARE TUTTE LE CONTRIBUION DELLE VARIABILI IN UN SINGOLO PLOT. TROPPI COLORI MISCHIATI. LASCIATO PERDERE
    # bias to obtained area...
    # l = np.array(l)
    # l[:, -1] = np.array(l[:, -1], dtype=np.float64)
    # min_ = np.min(np.array(l[:, -1], dtype=np.float64))
    # l[:, -1] = np.array(l[:, -1], dtype=np.float64) - min_ if min_ < 0 else l[:, -1]

    contributions_df = pd.DataFrame(l, columns=columns).sort_values(by=[str(selected_var_name), "features"],
                                                                    ascending=True).reset_index(
        drop=True)

    # opacity=0.6, smooth=True, line=False, interpolate='basis'
    chart = alt.Chart(contributions_df).mark_area(opacity=0.6, line=True, interpolate='basis').encode(
        x=str(selected_var_name) + str(":Q"),
        y=alt.Y("contributes", type="quantitative"),
        color="features:N",
        row="features:N"
    ).properties(
        height=100,
        width=700
    )

    return chart.interactive()


@st.experimental_memo(ttl=60 * 60)
def make_pyplot_bars_coef(coef, variables):
    """
    Same function as the method of Glime!
    """
    # TODO: quando converto la dashboard a Glime invece che mob,
    # questo plot lo richiamo direttamente utilizzando il metodo di Glime

    bars = pd.DataFrame(index=variables)
    bars["coef"] = coef
    bars['positive'] = bars["coef"] > 0

    # https://plotly.com/python-api-reference/generated/plotly.express.bar.html
    bars['positive'] = bars['positive'].map({True: 'positive', False: 'negative'})

    fig_px = px.bar(data_frame=bars, x="coef", y=bars.index.values.tolist(), labels={"y": "variables"},
                    color="positive",
                    color_discrete_map={
                        "negative": 'red',
                        "positive": 'green'
                    }).update_yaxes(categoryorder="total ascending")

    return fig_px


#
# @st.experimental_memo(ttl=60*60)
# def tridimensional_fig():
#     x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
#     y = x.copy().T
#     z = np.cos(x ** 2 + y ** 2)
#
#     fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
#     return fig


###multidim visualization

@st.experimental_memo(ttl=60 * 60)
def initial_nodes():
    return list()


@st.experimental_memo(ttl=60 * 60)
def get_selected_nodes(_model, datapoint, var):
    """
    Function that takes as input a datapoint (as an array of values),
    a variable name, which corresponds to the variable for the what-if analysis
    and the mob fitted model.
    Returns all the mob leaves that contain the datapoint (with the selected variable which is free to change value,
    spanning all the leaves we retrieve, when its value changes)

    Args:
        _model: mob fitted model
        datapoint: datapoint as an array
        var: selected variable for the what-if analysis

    Returns: list of nodes traversed by the what-if datapoint changing only the selected variable var
    """
    init_time = time.time()
    global_domain = _model.domain_dict
    node_list = [node for node in _model.leaves if
                 node.domain.domain_contains_without_var(datapoint, var, global_domain=global_domain)]
    end_time = time.time()
    return node_list, end_time - init_time


@st.cache(ttl=60 * 60)
def get_selected_nodes2(_model, datapoint, var):
    """
    Tested an implementation to exploit multithreading and speedup time computation.
    It does not work well...
    :param _model:
    :param datapoint:
    :param var:
    :return:
    """
    init_time = time.time()

    def valid_node(node, datapoint, var, _model):
        if node.domain.domain_contains_without_var(datapoint, var, global_domain=_model.domain_dict):
            return node

    valid_node_func = functools.partial(valid_node, datapoint=datapoint, var=var, _model=_model)

    def get_selected_nodes_inner(valid_node_func, leaves, workers):
        with ThreadPoolExecutor(workers) as pool:
            node_list = pool.map(valid_node_func, leaves)
            return [node for node in node_list if node]

    node_list = get_selected_nodes_inner(valid_node_func, _model.leaves, workers)
    end_time = time.time()
    return node_list, end_time - init_time


# TODO USA QUELLA DI VISUALIZATION
@st.experimental_memo(ttl=60 * 60)
def visualize_regressions_fixed_val(_nodes: list, var, what_if_datapoint, point, mode="html", path="..\\imgs\\regression.html",
                                    color_regr="blue", color_split="red", coef_bounds=(-5, 5)):
    """
    Create the Plot of Mob Linear Regressions of the leaves, for the What-if Analysis of a selected variable var


    Args:
        _model:  fitted mob model
        var: selected_variable name (for what-if)
        what_if_datapoint: array of a single datapoint. It is the what-if datapoint
        mode: if mode is "html", a browser window is opened and the plot is drawn in the browser.
              Otherwise, the plot is returned as output of the function
        path:
        color_regr:
        color_split:
        coef_bounds: tuple
                     y axis coordinates for picturing the vertical lines of the splits between mob leaves
                     if coef_bounds == (0, 0) --> no splitpoint bars in the figure
    Returns:
    :param plot_scaling:

    """

    what_if_datapoint = np.array(what_if_datapoint, dtype=np.float64)

    # List of figures, each one containing a Linear Regression of one leaf
    figures = []
    for leaf in _nodes:

        # retrieve the leaf bounds for the selected variable var
        bounds = leaf.domain[var].bounds

        # scrolling from the "leftest" node to the "rightest" node in a line given by the array

        # create two datapoints, with the same values of the what_if_datapoint, but the selected_variable takes
        # values as the left bound and right bound of its domain in the current leaf
        left_bound_datapoint = copy.copy(what_if_datapoint)
        left_bound_datapoint[var] = bounds[0]  # left bound for current leaf
        right_bound_datapoint = copy.copy(what_if_datapoint)
        right_bound_datapoint[var] = bounds[1]  # right bound for current leaf

        # mob predict on left_bound_datapoint and right_bound_datapoint using the current leaf regression
        y_left_bound_datapoint = leaf.regression.predict(left_bound_datapoint.reshape(1, -1))
        y_right_bound_datapoint = leaf.regression.predict(right_bound_datapoint.reshape(1, -1))
        y = (y_left_bound_datapoint, y_right_bound_datapoint)

        # create dataframe of the two boundary datapoints
        x_y = np.c_[np.array(bounds).reshape(-1, 1), np.array(y).reshape(-1, 1)]
        boundary_df = pd.DataFrame(x_y, columns=[str(var), "y"])

        # figure of the linear regression between the two points
        fig = alt.Chart(boundary_df).mark_line().encode(
            x=alt.X(str(var), type="quantitative", title=point["x"]),
            y=alt.Y("y", type="quantitative", title=point["y"]),
            color=alt.value(color_regr),
            tooltip=[str(var), "y"]
        )

        # plot the splitpoints as vertical lines
        if coef_bounds != (0, 0):
            ycut = np.array(coef_bounds) * 1.5
            xcut = [bounds[0], bounds[0]]
            xcut_ycut = np.c_[np.array(xcut).reshape(-1, 1), np.array(ycut).reshape(-1, 1)]
            df_cut = pd.DataFrame(xcut_ycut, columns=[str(var), "y"])
            cut = alt.Chart(df_cut).mark_line().encode(
                x=alt.X(str(var), type="quantitative"),
                y=alt.Y("y", type="quantitative"),
                color=alt.value(color_split)
            )
            figures.append(cut)

        figures.append(fig)

    chart = alt.layer(*figures)  # .interactive()

    if mode == "html":
        # chart.save(path)
        # webbrowser.open_new_tab(path)
        chart.show(open_browser=True)

    return chart


@st.experimental_memo
def global_importance_true_model(_model):
    coeffs_dict, true_model_fig = _model.global_importance(true_to="model", data=None, show=False)
    return true_model_fig


@st.experimental_memo
def global_importance_true_data(_model, data):
    coeffs_dict, true_data_fig = _model.global_importance(true_to="data", data=data, show=False)
    return true_data_fig


def check_performance(_glime):
    n_sobol_points = 2 ** _glime.n_sobol_points
    leaves = mob_utils.get_leaves(_glime.mob.tree)
    r2 = sum([leaf.score * leaf.n_samples / n_sobol_points for leaf in leaves])
    st.write(
        "Number of Sobol points = {:.0f}, Average R2 = {:.2f}, Requested Glime R2 = {:.2f}".format(n_sobol_points, r2,
                                                                                                   _glime.stopping_value))
    num_leaves = len(leaves)
    avg_leaf_size = n_sobol_points / num_leaves
    st.write(
        "Number of leaves created = {:.0f}, Average Leaf size = {:.1f}, Minimum requested size = {:.0f}".format(
            num_leaves, avg_leaf_size, _glime.minsplit))
    st.write("How many leaves have {:.0f}-{:.0f} Sobol points? {:.0f}".format(_glime.minsplit, _glime.minsplit + 5,
                                                                         sum([leaf.n_samples <= 20 for leaf in
                                                                              leaves])))
