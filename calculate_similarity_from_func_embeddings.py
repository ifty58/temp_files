from scipy import spatial
import math
import datetime
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

ARM_Emb_File = '/home/iftakhar/Ifty_All/function_comparison/v3_ARM_without_openssl_final_function_embeddings-tf_only.txt'
x86_Emb_File = '/home/iftakhar/Ifty_All/function_comparison/v3_x86_without_openssl_final_function_embeddings-tf_only.txt'
x86_emb_dict = {}
ARM_emb_dict = {}
from pyemd import emd
from sklearn.metrics import euclidean_distances


def calc_emd_alias_word_movers_distance(v_1, v_2):
    # W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
    v_1 = np.asarray(v_1, dtype=float)
    v_2 = np.asarray(v_2, dtype=float)

    v_1_mod = v_1.reshape(-1, 1)
    v_2_mod = v_2.reshape(-1, 1)

    D_ = euclidean_distances(v_1_mod, v_2_mod)
    # pyemd needs double precision input
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()  # just for comparison purposes
    # print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))
    return emd(v_1, v_2, D_)
    # return 1

def get_cosine_similarity(feature_vec_1, feature_vec_2):
    feature_vec_1 = np.asarray(feature_vec_1, dtype=np.float)
    feature_vec_2 = np.asarray(feature_vec_2, dtype=np.float)

    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

# FOR KL NAN ELIMINATED WHEN NORMALIZED
# FOR KL INF ELIMINATED WHEN 0 REPLACED BY VERY SMALL VALUE
# BUT RESULT DOES NOT IMPROVE
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    a = np.where(a != 0, a, 10E-300)
    b = np.where(b != 0, b, 10E-300)

    # return np.sum(np.where(a != 0, a * np.log(a / b), 0))
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def plot_curve(x, y, title):
    # x = df.ix[:, 0]
    # y = df.ix[:, 1]
    # naming the x axis
    plt.xlabel('function-count')
    # naming the y axis
    plt.ylabel('cosine similarity')
    # giving a title to my graph
    plt.title(title)
    # plotting the points
    plt.plot(x, y)
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    # function to show the plot
    plt.show()


def normalize_emb_list_for_KL(original_vals):
    max = np.max(original_vals)
    min = np.min(original_vals)
    scaled = np.array([(x - min) / (max - min) for x in original_vals])
    # print(scaled)
    return scaled
    # original_vals = [-23.5, -12.7, -20.6, -11.3, -9.2, -4.5, 2, 8, 11, 15, 17, 21]

    # get max absolute value
    # original_max = max([abs(val) for val in original_vals])
    #
    # # normalize to desired range size
    # new_range_val = 1
    # normalized_vals = [float(val) / original_max * new_range_val for val in original_vals]
    # return normalized_vals


with open(x86_Emb_File, 'r') as x86_file:
    for x86_line in x86_file:
        x86_line_details = x86_line.split(' ', 1)
        x86_file_name = x86_line_details[0]
        x86_emb = x86_line_details[1]
        # print('x86_file_name = ', x86_file_name)
        # print('x86_emb = ', x86_emb)
        x86_emb_dict[x86_file_name] = x86_emb
    # print('x86_emb_dict = ', x86_emb_dict)

with open(ARM_Emb_File, 'r') as ARM_file:
    for ARM_line in ARM_file:
        ARM_line_details = ARM_line.split(' ', 1)
        ARM_file_name = ARM_line_details[0]
        modified_ARM_file_name = ARM_file_name.replace('-ARM-',
                                                       '-')  # ARM_file_name.split('-ARM-')[0] + '-' + ARM_file_name.split('-ARM-')[1]
        ARM_emb = ARM_line_details[1]
        # print('x86_file_name = ', x86_file_name)
        # print('x86_emb = ', x86_emb)
        ARM_emb_dict[modified_ARM_file_name] = ARM_emb
    # print('ARM_emb_dict = ', ARM_emb_dict)

# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

# # below for same functions
sim_cosine_vals_X = []
sim_cosine_vals = []
count = 0
for file_name in x86_emb_dict:
    if file_name in ARM_emb_dict:
        # print('x86_emb_dict[file_name] = ', x86_emb_dict[file_name])
        x86_list = [float(i) for i in x86_emb_dict[file_name].split()]
        ARM_list = [float(i) for i in ARM_emb_dict[file_name].split()]
        print('file_name = ', file_name)
        #   for cosine similarity
        result = 1 - spatial.distance.cosine(x86_list, ARM_list)
        # result = get_cosine_similarity(x86_list, ARM_list)

        # for EMD/WMD
        # result = calc_emd_alias_word_movers_distance(x86_list, ARM_list)

        # dot(vector_a, vector_b, out = None)
        # result = np.dot(x86_list, ARM_list, out=None)

        # for KL divergence
        # result = KL(x86_list, ARM_list)
        # result = KL(normalize_emb_list_for_KL(x86_list), normalize_emb_list_for_KL(ARM_list))

        # wasserstein_distance
        # result = wasserstein_distance(ARM_list, x86_list)
        # print('result = ', result)
        sim_cosine_vals.append(result)
        count += 1
        sim_cosine_vals_X.append(count)
        # for i in x86_emb_dict[file_name].split():
        #     print('i = ', i)
        #     print(float(i))

        # x86_sum_emb = sum(x86_list)
        # ARM_sum_emb = sum(ARM_list)
        #
        # print('x86_sum_emb = ', x86_sum_emb)
        # print('ARM_sum_emb = ', ARM_sum_emb)
        # print('##################')
#         x86_minus_ARM_sum_emb = x86_sum_emb - ARM_sum_emb
#         diff_of_sum.append(x86_minus_ARM_sum_emb)
# sim_cosine_vals = set(sim_cosine_vals)
# sim_cosine_vals_X = set()
sim_cosine_vals = sorted(sim_cosine_vals, key=float)
# print('sim_cosine_vals = ', sim_cosine_vals)
# plot_curve(sim_cosine_vals_X, sim_cosine_vals, 'was.. Similar')

# dissim_count = 0
# dissim_cosine_vals_X = []

# Below for dissim all
# # max_dissim_count = 0
# dissim_cosine_vals = []
# for file_name_x86 in x86_emb_dict:
#     # if dissim_count > 2720:
#     #     break
#     for file_name_ARM in ARM_emb_dict:
#         file_name_ARM_modified = file_name_ARM.replace('-ARM-', '-')
#
#         if not file_name_x86 == file_name_ARM_modified:
#             x86_list = [float(i) for i in x86_emb_dict[file_name_x86].split()]
#             ARM_list = [float(i) for i in ARM_emb_dict[file_name_ARM_modified].split()]
#
#             print('file_name_x86 = ', file_name_x86)
#             print('file_name_ARM_modified = ', file_name_ARM_modified)
#             result = 1 - spatial.distance.cosine(x86_list, ARM_list)
#             # print('result = ', result)
#             dissim_cosine_vals.append(result)
#             dissim_count+=1
#             dissim_cosine_vals_X.append(dissim_count)
#             # break
#     # break
#             # for i in x86_emb_dict[file_name].split():
#             #     print('i = ', i)dissim_count
#             #     print(float(i))
#
#             # x86_sum_emb = sum(x86_list)
#             # ARM_sum_emb = sum(ARM_list)
#             #
#             # print('x86_sum_emb = ', x86_sum_emb)
#             # print('ARM_sum_emb = ', ARM_sum_emb)
#             # print('##################')
#             # x86_minus_ARM_sum_emb = x86_sum_emb - ARM_sum_emb
#             # print('x86_minus_ARM_sum_emb = ', x86_minus_ARM_sum_emb)
#             # diff_of_sum_dissim.append(x86_minus_ARM_sum_emb)
#
#             # break
# # diff_of_sum_dissim = set(diff_of_sum_dissim)
# # diff_of_sum_dissim = sorted(diff_of_sum_dissim, key = float)
# # print('diff_of_sum_dissim = ', diff_of_sum_dissim)
# dissim_cosine_vals = sorted(dissim_cosine_vals, key = float)
# plot_curve(dissim_cosine_vals_X, dissim_cosine_vals)

# Dissim Randomized for 7415 functions
dissim_count = 0
dissim_cosine_vals_X = []
dissim_cosine_vals = []
x86_keys = list(x86_emb_dict.keys())
random.shuffle(x86_keys)
for x86_key_filename in x86_keys:
    ARM_keys = list(ARM_emb_dict.keys())
    random.shuffle(ARM_keys)

    for ARM_key_filename in ARM_keys:
        # file_name_ARM_modified = ARM_key_filename.replace('-ARM-', '-')
        if not x86_key_filename == ARM_key_filename:
            x86_list = [float(i) for i in x86_emb_dict[x86_key_filename].split()]
            ARM_list = [float(i) for i in ARM_emb_dict[ARM_key_filename].split()]
            print(x86_key_filename)
            print(ARM_key_filename)

            #   for cosine similarity
            result = 1 - spatial.distance.cosine(x86_list, ARM_list)
            # result = get_cosine_similarity(x86_list, ARM_list)

            # for EMD/WMD
            # result = calc_emd_alias_word_movers_distance(x86_list, ARM_list)

            # dot(x86_list, ARM_list, out=None)
            # result = np.dot(x86_list, ARM_list, out=None)

            # for KL divergence
            # result = KL(x86_list, ARM_list)
            # result = KL(normalize_emb_list_for_KL(x86_list), normalize_emb_list_for_KL(ARM_list))

            # wasserstein_distance
            # result = wasserstein_distance(ARM_list, x86_list)

            # print('result = ', result)
            dissim_cosine_vals.append(result)
            dissim_count += 1
            dissim_cosine_vals_X.append(dissim_count)
            break

    # if dissim_count > 7415:
    #     break
dissim_cosine_vals = sorted(dissim_cosine_vals, key=float)
# plot_curve(dissim_cosine_vals_X, dissim_cosine_vals, 'was... dissimilar')
y_test, y_pred_proba = [], []
# USING COSINE SIMILARITY
# for i in range(22):
#     threshold = 0.0 + i * 0.025
#     for val in dissim_cosine_vals:
#         if val < threshold:
#             y_test.append(0)
#             y_pred_proba.append(0)
#         else:
#             y_test.append(0)
#             y_pred_proba.append(1)
#
#     for val in sim_cosine_vals:
#         if val >= threshold:
#             y_test.append(1)
#             y_pred_proba.append(1)
#         else:
#             y_test.append(1)
#             y_pred_proba.append(0)
#     auc = metrics.roc_auc_score(y_test, y_pred_proba)
#     print('threshold = ', threshold, 'auc = ', auc)

acu_vals_Y = []
threshold_vals_X = []

#  BELOW POCESS DOES NOT GIVE SMOOTH CURVE**********
# USING KL DIVERGENCE
# for i in range(22):
# threshold_simi_low = -0.1
# threshold_simi_up = 0.1
# for i in range(40):
# threshold = 0.62 #0.6 + i * 0.02 # FOR KL Final
# threshold = 0.001 + i * 0.0002 # FOR was....
# threshold = 0.004 #0.001 + i * 0.0002 # FOR was.... FINAL
# for val in dissim_cosine_vals:
#     if val < threshold:
#         y_test.append(0)
#         y_pred_proba.append(1)
#     else:
#         y_test.append(0)
#         y_pred_proba.append(0)
#
# for val in sim_cosine_vals:
#     # if val >= threshold_simi_low and val <= threshold_simi_up:
#     if val >= threshold:
#         y_test.append(1)
#         y_pred_proba.append(0)
#     else:
#         y_test.append(1)
#         y_pred_proba.append(1)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# # print('auc = ', auc)
# print('threshold = ', threshold, 'auc = ', auc)
# acu_vals_Y.append(auc)
# threshold_vals_X.append(threshold)

# plot_curve(threshold_vals_X, acu_vals_Y, 'AUC vs Threshold')
# FOR SMOOTH CURVE
# FOR wasserstein_distance........... AND KL DIVERGENCE
# DISSIMILAR = 1
# SIMIALR = 0

# cosine and dot FINAL Vals
dissimilar_val_default = 0  # 0.004 + delta
similar_val_default = 1  # 0.004 - delta


delta = 0.0005

# wasserstein_distance FINAL Vals // also for WMD //KL
# dissimilar_val_default = 1  # 0.004 + delta
# similar_val_default = 0  # 0.004 - delta


for val in dissim_cosine_vals:
    if math.isnan(val):
        print('nan')
    elif math.isinf(val):
        print('inf')
    else:
        y_test.append(dissimilar_val_default)
        y_pred_proba.append(val)

for val in sim_cosine_vals:
    if math.isnan(val):
        print('nan')
    elif math.isinf(val):
        print('inf')
    else:
        y_test.append(similar_val_default)
        y_pred_proba.append(val)

# auc = metrics.roc_auc_score(y_test, y_pred_proba)*100
# print('auc = ', auc)
# print('threshold = ', threshold, 'auc = ', auc)
# acu_vals_Y.append(auc)
# threshold_vals_X.append(threshold)


# # create ROC curve
# curve_title = 'wasserstein_distance'
curve_title = 'KL Divergence'

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr) * 100
plt.plot(fpr, tpr, label="AUC=" + str(roc_auc) + '%')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title(curve_title)
plt.legend(loc=4)
plt.show()
