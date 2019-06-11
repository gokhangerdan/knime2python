import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xmltodict
import numpy as np
from sklearn.cluster import KMeans
import operator
import sys
import warnings
import re
import statistics


warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True


class KnimeNode:
    def __init__(self, func, params=None, input_node=None):
        self.output_table = func(params, input_node)
        self.func = func
        self.params = params
        self.input_node = input_node
        try:
            self.sample = self.output_table.head()
        except:
            self.sample = None
        try:
            self.rows = len(self.output_table)
        except:
            self.rows = None
        try:
            self.columns = self.output_table.keys(), len(self.output_table.keys())
        except:
            self.columns = None
        try:
            self.clusters = self.output_table[0]
        except:
            self.cluster = None
        try:
            self.model = self.output_table[1]
        except:
            self.model = None


class NodeRepository:
    class IO:
        class Read:
            @staticmethod
            def csv_reader(params, input_node):
                return pd.read_csv(params["input_location"])

            @staticmethod
            def pmml_reader(params, input_node):
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
        class Column:
            class ConvertAndReplace:
                @staticmethod
                def column_rename(params, input_node):
                    input_table = input_node.output_table
                    return input_table.rename(index=str, columns=params["change_columns"])

            class Filter:
                @staticmethod
                def column_filter(params, input_node):
                    return input_node.output_table.filter(items=params["include"])

            class SplitAndCombine:
                @staticmethod
                def column_combiner(params, input_node):
                    input_table = input_node.output_table
                    delimiter = params["delimiter"]
                    new_column = params["name_of_appended_column"]
                    include = params["include"]
                    input_table[new_column] = input_table[include].apply(
                        lambda x: delimiter.join(x), axis=1)
                    return input_table

            class Transform:
                @staticmethod
                def missing_value(params, input_node):
                    table = input_node.output_table
                    option = params["option"]
                    if option == "default":
                        string = params["string"]
                        value = params["value"]
                        if string == "fix_value":
                            return table.fillna(value)
                    elif option == "column_settings":
                        include = params["include"]
                        value = params["value"]
                        if value == "previous_value":
                            previous_value = {}
                            for column in include:
                                previous_value[column] = None
                            new_rows = []
                            for index, row in table.iterrows():
                                new = {}
                                for column in table.keys():
                                    if column in include:
                                        if not pd.isnull(row[column]):
                                            previous_value[column] = row[column]
                                            new[column] = row[column]
                                        else:
                                            row[column] = previous_value[column]
                                            new[column] = row[column]
                                    else:
                                        new[column] = row[column]
                                new_rows.append(new)
                            return pd.DataFrame(new_rows)

        class Row:
            class Filter:
                @staticmethod
                def row_filter(params, input_node):
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
                            if "pattern" not in params.keys() and "regular_expression" in params.keys():
                                def match_regex(x):
                                    res = re.match(
                                        params["regular_expression"], x)
                                    if res:
                                        return True
                                    else:
                                        return False
                                input_table[column_to_test+"_match_regex"] = input_table[column_to_test].apply(
                                    lambda x: match_regex(x))
                                return input_table[input_table[column_to_test+"_match_regex"] == True].drop(columns=[column_to_test+"_match_regex"])
                            else:
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
                            if "pattern" not in params.keys() and "regular_expression" in params.keys():
                                def match_regex(x):
                                    res = re.match(
                                        params["regular_expression"], x)
                                    if res:
                                        return True
                                    else:
                                        return False
                                input_table[column_to_test+"_match_regex"] = input_table[column_to_test].apply(
                                    lambda x: match_regex(x))
                                return input_table[input_table[column_to_test+"_match_regex"] == False].drop(columns=[column_to_test+"_match_regex"])
                            else:
                                pattern = params["pattern"]
                                return input_table[input_table[column_to_test] != pattern]

            class Transform:
                @staticmethod
                def concatenate(params, input_node):
                    if params["column_handling"] == "use_union_of_columns":
                        return pd.concat([input_node[x].output_table for x in input_node.keys()])

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

                    def concatenate(x):
                        return ', '.join(list([str(i) for i in x]))

                    def maximum(x):
                        return max(list(x))

                    def minimum(x):
                        return min(list(x))

                    def mean(x):
                        return statistics.mean(list(x))

                    def median(x):
                        return statistics.median(list(x))

                    def standard_deviation(x):
                        data_list = list(x)
                        if len(data_list) == 1:
                            return 0
                        else:
                            return statistics.stdev(data_list)

                    aggregation = {
                        "list": list_agg,
                        "set": set_agg,
                        "unique_count": unique_count,
                        "append_elements": append_elements,
                        "union_count": union_count,
                        "concatenate": concatenate,
                        "maximum": maximum,
                        "minimum": minimum,
                        "mean": mean,
                        "median": median,
                        "standard_deviation": standard_deviation,
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
                    table = input_node.output_table
                    ungroup_columns = params["include"]
                    new_rows = []
                    for index, row in table.iterrows():
                        max_length = max([len(cell)
                                          for cell in row if type(cell) is list])
                        for new_row in range(max_length):
                            new = {}
                            for column in table.keys():
                                if column not in ungroup_columns or type(row[column]) is not list:
                                    new[column] = row[column]
                                else:
                                    try:
                                        new[column] = row[column][new_row]
                                    except IndexError:
                                        new[column] = None
                            new_rows.append(new)
                    return pd.DataFrame(new_rows)

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

                    def rule_apply(row, expression, operator_dict=operator_dict):
                        true_false = False
                        for i in expression:
                            if operator_dict[i[1]](row[i[0]], i[2]):
                                true_false = True
                                return i[4]
                        if not true_false:
                            return None
                    input_table[append_column] = input_table.apply(
                        lambda row: rule_apply(row, expression), axis=1)
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
                    cluster_col_name = params["new_column"]
                    labels = pd.DataFrame(labels, columns=[cluster_col_name])
                    elements = list(set(labels[cluster_col_name].tolist()))
                    elements.sort()
                    labels = labels.replace(
                        elements, ["cluster_"+str(x) for x in elements])
                    return pd.concat([port_1.reset_index(), labels.reset_index()], axis=1).drop(["index"], axis=1)

    class Scripting:
        class Python:
            @staticmethod
            def python_source(params, input_node):
                script = __import__(params["script"])
                return script.foo()

            @staticmethod
            def python_script(params, input_node):
                input_table = input_node.output_table
                script = __import__(params["script"])
                return script.foo(input_table)
