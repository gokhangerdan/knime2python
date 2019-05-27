import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xmltodict
import numpy as np
from sklearn.cluster import KMeans
import operator
import sys
import warnings


warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True


class KnimeNode:
    """
    class knime2python.KnimeNode(func=None, params=None, input_node=None)

    Parameters:
    func: NodeRepository function (e.g. NodeRepository.IO.Read.csv_reader)
    params: dict, default None
    input_node: knime2python.KnimeNode, default None
    """

    def __init__(self, func, params=None, input_node=None):
        self.output_table = func(params, input_node)


class NodeRepository:
    """
    knime2python.NodeRepository

    .IO
        .Read
            .csv_reader
            .pmml_reader
    .Manipulation
        .Column
            .ConvertAndReplace
                .column_rename
            .Filter
                .column_filter
        .Row
            .Filter
                .row_filter
            .Transform
                .group_by
                .ungroup
            .Other
                .rule_engine
    .OtherDataTypes
        .TimeSeries
            .Manipulate
                .date_and_time_difference
            .Transform
                .string_to_date_and_time
    .Analytics
        .Mining
            .Clustering
                .cluster_assigner
    .Scripting
        .Python
            .python_script
    """
    class IO:
        """
        knime2python.NodeRepository.IO

        .Read
            .csv_reader
            .pmml_reader
        """
        class Read:
            """
            knime2python.IO.Read

            .csv_reader
            .pmml_reader
            """
            @staticmethod
            def csv_reader(params, input_node):
                """
                knime2python.IO.Read.csv_reader

                Parameters:
                input_location: str
                """
                return pd.read_csv(params["input_location"])

            @staticmethod
            def pmml_reader(params, input_node):
                """
                knime2python.IO.Read.pmml_reader

                Parameters:
                input_location: str
                """
                xml_file = open(params["input_location"], "r").read()
                a = dict(xmltodict.parse(xml_file))
                clusters = a["PMML"]["ClusteringModel"]["Cluster"]
                fields = a["PMML"]["ClusteringModel"]["ClusteringField"]
                if type(fields) == list:
                    fields = [x["@field"] for x in fields]
                else:
                    fields = [fields["@field"]]
                x = np.array(
                    [[eval(x) for x in i["Array"]["#text"].split(" ")] for i in clusters])
                kmeans = KMeans(
                    n_clusters=len(x),
                    init=x,
                    n_init=1,
                    random_state=1
                )
                return fields, kmeans

    class Manipulation:
        """
        knime2python.NodeRepository.Manipulation

        .Column
            .ConvertAndReplace
                .column_rename
            .Filter
                .column_filter
        .Row
            .Filter
                .row_filter
            .Transform
                .group_by
                .ungroup
            .Other
                .rule_engine
        """
        class Column:
            """
            knime2python.NodeRepository.Manipulation.Column

            .ConvertAndReplace
                .column_rename
            .Filter
                .column_filter
            """
            class ConvertAndReplace:
                """
                knime2python.NodeRepository.Manipulation.ConvertAndReplace

                .column_rename
                """
                @staticmethod
                def column_rename(params, input_node):
                    input_table = input_node.output_table
                    return input_table.rename(index=str, columns=params["change_columns"])

            class Filter:
                """
                knime2python.NodeRepository.Manipulation.Filter

                .column_filter
                """
                @staticmethod
                def column_filter(params, input_node):
                    """
                    knime2python.NodeRepository.Manipulation.Filter.column_filter

                    Parameters:
                    include: list (of strings)
                    """
                    return input_node.output_table.filter(items=params["include"])

        class Row:
            """
            knime2python.NodeRepository.Manipulation.Row

            .Filter
                .row_filter
            .Transform
                .group_by
                .ungroup
            .Other
                .rule_engine
            """
            class Filter:
                """
                knime2python.NodeRepository.Manipulation.Row.Filter

                .row_filter
                """
                @staticmethod
                def row_filter(params, input_node):
                    """
                    knime2python.NodeRepository.Manipulation.Row.Filter.row_filter

                    Parameters:
                    filter_criteria: str (2 options):
                        include_rows_by_attribute_value
                        exclude_rows_by_attribute_value
                    column_to_test: str
                    matching_criteria: str (3 options)
                        only_missing_values_match
                        use_range_checking (require additional parameters):
                            lower_bound: int
                            upper_bound: int
                        use_pattern_matching (require additional parameters):
                            pattern: str
                    """
                    input_table = input_node.output_table
                    filter_criteria = params["filter_criteria"]
                    column_to_test = params["column_to_test"]
                    matching_criteria = params["matching_criteria"]
                    if filter_criteria == "include_rows_by_attribute_value":
                        if matching_criteria == "only_missing_values_match":
                            return input_table[input_table[column_to_test].isnull()]
                        elif matching_criteria == "use_range_checking":
                            lower_bound = params["lower_bound"]
                            upper_bound = params["upper_bound"]
                            if lower_bound != None and upper_bound != None:
                                return input_table[input_table[column_to_test] >= lower_bound & (input_table[column_to_test] <= upper_bound)]
                            elif lower_bound != None and upper_bound == None:
                                return input_table[input_table[column_to_test] >= lower_bound]
                            elif lower_bound == None and upper_bound != None:
                                return input_table[input_table[column_to_test] <= upper_bound]
                        elif matching_criteria == "use_pattern_matching":
                            pattern = params["pattern"]
                            return input_table[input_table[column_to_test] == pattern]
                    elif filter_criteria == "exclude_rows_by_attribute_value":
                        if matching_criteria == "only_missing_values_match":
                            return input_table[input_table[column_to_test].notnull()]
                        elif matching_criteria == "use_range_checking":
                            lower_bound = params["lower_bound"]
                            upper_bound = params["upper_bound"]
                            if lower_bound != None and upper_bound != None:
                                return input_table[input_table[column_to_test] < lower_bound | (input_table[column_to_test] > upper_bound)]
                            elif lower_bound != None and upper_bound == None:
                                return input_table[input_table[column_to_test] < lower_bound]
                            elif lower_bound == None and upper_bound != None:
                                return input_table[input_table[column_to_test] > upper_bound]
                        elif matching_criteria == "use_pattern_matching":
                            pattern = params["pattern"]
                            return input_table[input_table[column_to_test] != pattern]

            class Transform:
                """
                knime2python.NodeRepository.Manipulation.Row.Transform

                .group_by
                .ungroup
                """
                @staticmethod
                def group_by(params, input_node):
                    input_table = input_node.output_table

                    def list_agg(x):
                        return list(x)

                    def set_agg(x):
                        return list(set(x))

                    def unique_count(x):
                        return len(list(set(x)))

                    def append_elements(x):
                        return sum(list(x), [])

                    def union_count(x):
                        return len(sum(list(x), []))
                    aggregation = {
                        "list": list_agg,
                        "set": set_agg,
                        "unique_count": unique_count,
                        "append_elements": append_elements,
                        "union_count": union_count,
                    }
                    manuel_aggregation = params["manuel_aggregation"]
                    new_manuel_aggregation = {}
                    drop_items = []
                    for i in manuel_aggregation.keys():
                        if type(manuel_aggregation[i]) == list:
                            drop_items.append(i)
                            for j in manuel_aggregation[i]:
                                input_table[i+"_"+j] = input_table[i]
                                new_manuel_aggregation[i +
                                                       "_"+j] = aggregation[j]
                        else:
                            new_manuel_aggregation[i] = aggregation[manuel_aggregation[i]]
                    input_table = input_table.drop(drop_items, axis=1)
                    input_table = input_table.groupby(
                        params["group_columns"], as_index=False)
                    return input_table.agg(new_manuel_aggregation)

                @staticmethod
                def ungroup(params, input_node):
                    input_table = input_node.output_table
                    include = params["include"]
                    new_dataframe = []
                    for i in range(len(input_table)):
                        n = len(input_table[include[0]][i])
                        for j in range(n):
                            new_row = {}
                            for k in input_table.keys():
                                if type(input_table[k][i]) == list:
                                    new_row[k] = input_table[k][i][j]
                                else:
                                    new_row[k] = input_table[k][i]
                            new_dataframe.append(new_row)
                    return pd.DataFrame(new_dataframe)

            class Other:
                @staticmethod
                def rule_engine(params, input_node):
                    input_table = input_node.output_table
                    expression = params["expression"]
                    append_column = params["append_column"]
                    operator_dict = {
                        "=": operator.eq,
                        "!=": operator.ne,
                        ">": operator.gt,
                        "<": operator.lt,
                        "<=": operator.le,
                        ">=": operator.ge,
                    }
                    for i in expression:
                        input_table.loc[operator_dict[i[1]](
                            input_table[i[0]], i[2]), append_column] = i[4]
                    return input_table

    class OtherDataTypes:
        class TimeSeries:
            class Transform:
                @staticmethod
                def string_to_date_and_time(params, input_node):
                    input_table = input_node.output_table
                    include = params["include"]
                    format_ = params["format"]
                    for col_name in include:
                        input_table[col_name] = pd.to_datetime(
                            input_table[col_name], format=format_)
                    return input_table

            class Manipulate:
                @staticmethod
                def date_and_time_difference(params, input_node):
                    input_table = input_node.output_table
                    base_column = params["base_column"]
                    calculate_difference_to = params["calculate_difference_to"]
                    output_options = params["output_options"]
                    new_column_name = params["new_column_name"]

                    if output_options == "granularity":
                        granularity = params["granularity"]

                    if calculate_difference_to == "second_column":
                        second_column = params["second_column"]
                        input_table[new_column_name] = input_table.apply(lambda row: getattr(relativedelta(
                            row[second_column], row[base_column]), granularity) if not pd.isna(row[base_column]) else None, axis=1)
                        return input_table

    class Analytics:
        class Mining:
            class Clustering:
                @staticmethod
                def cluster_assigner(params, input_node):
                    port_0 = input_node["port_0"].output_table
                    port_1 = input_node["port_1"].output_table
                    labels = port_0[1].fit(
                        port_1.as_matrix(columns=port_0[0])).labels_
                    cnt = []
                    for i in port_1.keys():
                        if i != "Cluster" and "Cluster" in i:
                            cnt.append(i)
                    cluster_col_name = "Cluster_" + str(len(cnt))
                    labels = pd.DataFrame(labels, columns=[cluster_col_name])
                    elements = list(set(labels[cluster_col_name].tolist()))
                    elements.sort()
                    labels = labels.replace(
                        elements, ["cluster_"+str(x) for x in elements])
                    return pd.concat([port_1.reset_index(), labels.reset_index()], axis=1).drop(["index"], axis=1)

    class Scripting:
        class Python:
            @staticmethod
            def python_script(params, input_node):
                input_table = input_node.output_table
                script = __import__(params["script"])
                return script.foo(input_table)
