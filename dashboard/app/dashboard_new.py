from utils import *
import copy

import sys
import pickle
import time
import tensorflow as tf
import zipfile
import tempfile
import os
import keras

sys.path.append('..\\..\\helpers')
sys.path.append('..\\..\\pymob\\mob')


### page configuration
st.set_page_config(
    page_title="🌎",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'http://bitly.com/98K8eH',  #TODO: aggiungere veri format per help e bug
        'Report a bug': "http://bitly.com/98K8eH",
        'About': ""
    }
)

# helpful to show the tooltip when you widen the altair plots to full screen (without this tooltip doesn't work)
# https://discuss.streamlit.io/t/tool-tips-in-fullscreen-mode-for-charts/6800/9
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)

# to set the font size for the expander text
st.markdown(
    """
<style>
.streamlit-expanderHeader {
    font-size: x-medium;
}
</style>
""",
    unsafe_allow_html=True,
)

# set the color for higlighting text or points
color = "orange"


###composing###
top = st.container()
cont1, cont2 = st.container(), st.container()
sidebar = st.sidebar
bottom = st.container()



with sidebar:
    uploaded_file = st.file_uploader("Upload dataset", type="csv")
    uploaded_model = st.file_uploader("Upload model", type="pkl")
    # Dialog box asking to load Dataset and Explanation
    if uploaded_file is None and uploaded_model is None:
        "**Please Upload the Dataset and the GlobalLIME Explanation**"
    elif uploaded_file is None:
        "**Please Upload the Dataset**"
    elif uploaded_model is None:
        "**Please Upload the GlobalLIME Explanation**"
    # Upload dataset
    if uploaded_file is not None:
        uploaded_dataframe = pd.read_csv(uploaded_file)
    else:
        st.stop()


    #Upload explanation
    if uploaded_model is not None:
        # # IN STREAMLIT JUST PICKLE_LOAD, IN PYTHON I SHOULD OPEN THE PICKLE FILE BEFORE LOADING
        # with open(uploaded_model, "rb") as pickle_file:
        #     gleams = pickle.load(pickle_file)
        gleams = pickle.load(uploaded_model)
        if not gleams.is_fitted_:
            "**Please upload a fitted Explanation Model**"
        mob = gleams.mob

        if not gleams.predict_function:
            nn_model_file = st.file_uploader('TF.Keras predict function(.h5py.zip)', type='zip')
            if nn_model_file is None:
                "**Please Upload the Keras/Tensorflow model**"
                st.stop()
            else:
                # let the user choose a Keras model hdf5.zip file to upload in streamlit
                # inspired by: https://discuss.streamlit.io/t/upload-keras-models-or-pickled-files/2246/2
                myzipfile = zipfile.ZipFile(nn_model_file)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    myzipfile.extractall(tmp_dir)
                    root_folder = myzipfile.filename[:-4]  # e.g. "model.h5py" (we remove the .zip)
                    model_dir = os.path.join(os.path.normpath(tmp_dir), os.path.normpath(root_folder))
                    # st.info(f'trying to load model from tmp dir {model_dir}...')
                    nn_model = tf.keras.models.load_model(model_dir)
                    gleams.predict_fuction = nn_model.predict

                # nn_model = tf.keras.models.load_model(nn_model_file)
                # gleams.predict_fuction = nn_model.predict

    else:
        st.stop()

    # save useful stuff from gleams
    # independent variables and target variable
    variables = gleams.variable_names
    variable_ids = {var_id:var_name for var_id,var_name in enumerate(variables)}
    domain_dict_ids = gleams.domain_dict
    domain_dict_namevars = {variable_ids[k]:v for k,v in domain_dict_ids.items()}

    # find out the independent variable from the uploaded dataframe
    target_set = set(uploaded_dataframe.columns) - set(variables)
    try:
        target_variable = target_set.pop()
        if len(target_set):
            "**the Dataset contains additional variables that are not in Explanation model.\n Fix the dataset and reload it**"
            st.stop()
    except KeyError:
        "**No Target Variable in the Dataset.\n Fix the dataset and reload it**"
        st.stop()


    "---"

    # other variables value
    with sidebar.expander("Single variable What-if Scenario"):

        # initialize objects to be memoized, to allow clearing them when changing selected_var or unit
        nodes = list()
        subset_df_synth = list()
        subset_df_orig = list()

        # store selected row
        unit_list = ["Unit " + str(unit_number) for unit_number in range(len(uploaded_dataframe))]
        selected_row = int(st.selectbox("Select dataframe row", unit_list,
                                        on_change=nodes.clear()).split(" ")[1])
        selected_row_data = uploaded_dataframe.iloc[[selected_row]]
        selected_row_data_x = selected_row_data[variables]
        selected_row_data_y = selected_row_data[target_variable]
        # dictionary of the selected datapoint, of the shape {name_var:value}
        selected_datapoint_dict = {v: selected_row_data[v].to_numpy()[0] for v in variables}
        selected_datapoint_array = np.array(list(selected_datapoint_dict.values()))

        # store independent variable for 1d what-if analysis
        selected_var = st.selectbox('Select x variable', variables, on_change=nodes.clear())
        selected_var_id = list(variable_ids.keys())[list(variable_ids.values()).index(selected_var)]

        # slider for the what-if analysis in 1 variable
        min_, max_ = domain_dict_namevars[selected_var]
        width_x = np.abs(max_ - min_)
        slider_var = st.slider('Select a value for the variable', float(min_), float(max_), float(selected_datapoint_dict[selected_var]),
                               step=width_x / 100)
        altair_pad_x = width_x / 1000
        altair_scale_x = alt.Scale(domain=(min_ - altair_pad_x, max_ + altair_pad_x))

        #"---"

        # store what-if row
        whatif_row_data = copy.copy(selected_row_data)
        whatif_row_data[[selected_var]] = slider_var
        whatif_datapoint_dict = copy.copy(selected_datapoint_dict)
        whatif_datapoint_dict[selected_var] = slider_var
        whatif_datapoint_array = np.array(list(whatif_datapoint_dict.values()))

        # local coeffs of what-if datapoint
        whatif_node_coeffs = get_coefficients(mob, whatif_datapoint_array.reshape(1, -1))
        # local coef of the of what-if datapoint for the selected_var
        whatif_selected_var_coef = whatif_node_coeffs[selected_var_id]

        #what-if datapoint mob prediction
        whatif_mob_pred = mob.predict(whatif_datapoint_array.reshape(1, -1))[0]

        # delta of selected_var value between what-if datapoint and selected datapoint
        delta_xval = "{:.2f}".format(slider_var - selected_datapoint_dict[selected_var])
        selected_datapoint_mob_pred = mob.predict(selected_row_data_x)[0]
        delta_mob_pred = "{:.2f}".format(- selected_datapoint_mob_pred + whatif_mob_pred)

with cont1:

    st.markdown("**Explanation Model Details**")
    with st.expander("Explanation Model info"):
        check_performance(gleams)

    st.markdown("**Visualize Data**")

    with st.expander("Selected row to be explained"):
        st.dataframe(selected_row_data.style.format(precision=2))

    with st.expander("What-If scenario (changed only 1 value of the selected row)"):
        st.dataframe(highlight_cell_in_df(whatif_row_data, value_to_highlight=slider_var, selected_var=selected_var))

    "---"
    with st.expander("Original ML function & Gleams Explainable Model"):
        st.caption(""" \n Draw the original (synthetic) points belonging to the leaves concerned with the 1d Global What-if Scenario
(leaves containing the selected datapoint when changing the what-if variable over all its range)""")
        orig_synth = st.selectbox("Display Original or Synthetic (Sobol) points?", ["original","synthetic"])

        point = {"x": selected_var, "x_val": slider_var, "y": target_variable, "y_val": whatif_mob_pred}
        columns = {"x": selected_var, "y": target_variable} # axes variable names
        tooltip = [selected_var, target_variable] # info for mouse hovering
        # dictionary with info about the what-if datapoint

        #Get all leaves containing the selected datapoint (selected variable ranging all over its domain)
        nodes,nodes_time = get_selected_nodes(mob, selected_datapoint_array, selected_var_id)
        st.write("Time taken to retrieve the nodes: {} s".format(nodes_time))
        r2_list_nodes_w1d = [node.score for node in nodes]
        domain_list_nodes_w1d = [node.domain[selected_var_id] for node in nodes]
        npoints_list_nodes_w1d = [node.n_samples for node in nodes]

        #select all points contained in the list of nodes
        if orig_synth == "original":
            subset_df_orig, df_time = get_original_points(uploaded_dataframe, tuple(nodes))
            st.write("Time taken to retrieve the points: {} s".format(df_time))
            # this is a check to ensure that the subset_df changes when changing nodes
            st.write("Number of displayed units: {}".format(subset_df_orig.shape[0]))
            chart_orig = copy.deepcopy(make_altair_plot(subset_df_orig, point, columns=columns, tooltip=tooltip,
                                               coef=whatif_selected_var_coef))
            mob_fig = copy.deepcopy(
                visualize_regressions_fixed_val(nodes, var=selected_var_id, what_if_datapoint=selected_datapoint_array,
                                                point=point, mode="none", color_regr="green", coef_bounds=(0, 0)))
            chart_orig_final = alt.layer(chart_orig, mob_fig)

            st.altair_chart(chart_orig_final, use_container_width=True)

        elif orig_synth == "synthetic":
            subset_df_synth, df_time = get_synthetic_points(tuple(nodes), variables + [target_variable, ])
            st.write("Time taken to retrieve the points: {} s".format(df_time))

            # this is a check to ensure that the subset_df changes when changing nodes
            st.write("Number of displayed units: {}".format(subset_df_synth.shape[0]))

            chart_synth = copy.deepcopy(make_altair_plot(subset_df_synth, point, columns=columns, tooltip=tooltip,
                                               coef=whatif_selected_var_coef))
            mob_fig = copy.deepcopy(
                visualize_regressions_fixed_val(nodes, var=selected_var_id, what_if_datapoint=selected_datapoint_array,
                                                point=point, mode="none", color_regr="green", coef_bounds=(0, 0)))
            chart_synth_final = alt.layer(chart_synth, mob_fig)

            st.altair_chart(chart_synth_final, use_container_width=True)


        st.write("Leaves domains retrieved for the 1D What-if Analysis: {}".format(domain_list_nodes_w1d))
        st.write("Leaves R2 retrieved for the 1D What-if Analysis: {}".format(r2_list_nodes_w1d))
        st.write("number of points in Leaves retrieved for the 1D What-if Analysis: {}".format(npoints_list_nodes_w1d))

    "---"

    st.markdown("**Global What-If Scenarios**")

    with st.expander("Single Variable, What-if Scenario (governed from the sidebar)"):
        c1, c2 = st.columns(2)

        #TODO: change local_importance in local_what_if method, when we admit also categorical variables
        expl_dict, local_importance_fig = gleams.local_importance(whatif_datapoint_array, show=False)
        c1.plotly_chart(local_importance_fig, use_container_width=True)

        # metric is an object containing (label, value, delta (with arrow))
        # the values and deltas for these metrics where calculated at the beginning, in the sidebar
        c2.metric(label=f"Value of variable {selected_var}", value="{:.2f}".format(slider_var), delta=delta_xval)
        c2.metric(label="y explanation", value="{:.2f}".format(whatif_mob_pred), delta=delta_mob_pred)
        c2.metric(label="local coefficient", value="{:.2f}".format(whatif_selected_var_coef))


    with st.expander("All variables, What-if Scenario"):

        # basically, in one line we require to draw the sliders in the dashboard and at the same time,
        # we initialize the dictionary associating at each variable its value in the slider
        # (by default slider_value is the the minimum of the domain)
        # when the user changes the value using the slider, the dictionary changes values accordingly

        # create a dictionary like {name_var: slider_value}
        whatif_dict = {v: st.slider(f"{v}", min_value=domain_dict_namevars[v][0],
                                    max_value=domain_dict_namevars[v][1],
                                    value=float(selected_datapoint_dict[v])) for v in variables}
        whatif_datapoint_array = np.array([[v for _, v in whatif_dict.items()]])

        val = mob.predict(whatif_datapoint_array)[0]
        whatif_selected_var_coef = get_coefficients(mob, whatif_datapoint_array)

        # create the barplot for the leaf of the global_whatif_datapoint
        bars = make_pyplot_bars_coef(whatif_selected_var_coef, variables)

        st.plotly_chart(bars, use_container_width=True)

        # print the mob prediction value
        f"Explanation Predicted Value: {val}"

    "---"

    st.markdown("**Global Feature Importance**")

    with st.expander("True to the Model"):
        init_time = time.time()
        true_model_fig = global_importance_true_model(gleams)
        st.plotly_chart(true_model_fig, use_container_width=True)

    with st.expander("True to the Data"):
        true_data_fig = global_importance_true_data(gleams, data=uploaded_dataframe)
        st.plotly_chart(true_data_fig, use_container_width=True)
        end_time = time.time()
        st.write("TIme for the two Feat Imp figures: {} s".format(end_time-init_time))



