from fitness import *
from params import params as p


class Callbacks:
    def SavePopulation_bests(folder = "../results/" + p["input_name"] + "/tmp/", save_gen_step = 1, file_prefix = 'population_bests'):
        def func(generation_number, report_list, last_population, last_scores):
            bests = {'generation':[],'individual':[], 'fitness_train':[]}
            bests_metrics = {'y_pred_prob':[], 'y_real':[],
                             'loss': [], 'tp': [], 'fp': [], 'tn': [], 'fn': [],
                             'precision': [], 'recall': [], 'auc': [], 'prc': [], 'accuracy': []}
            bests_metrics_test = {'fitness_test':[], 'loss_test': [], 'tp_test': [], 'fp_test': [], 'tn_test': [], 'fn_test': [],
                                  'precision_test': [], 'recall_test': [], 'auc_test': [], 'prc_test': [], 'accuracy_test': []}
            if generation_number % save_gen_step != 0:
                return
            # Position best individual train fitness 
            best_ind = min(abs(last_scores)) if p["fitness_metric"] in ["loss", "fn", "fp"] else max(abs(last_scores))
            index_best = list(last_scores).index(best_ind)
            best_weights = list(last_population)[index_best]
            # Fitness test 
            X_train, y_train, X_test, y_test = LoadData().get_model_data()
            fitness_test = AnnModel(best_weights).predict_test(X_test, y_test)
            # Save metrics for fitness test
            for idx,i in enumerate(bests_metrics_test):
                bests_metrics_test[i] += [fitness_test[idx]]
            # Select best metric from all individuals of a population
            npz = np.load(folder+'metrics_tmp_best_population.npz', allow_pickle=True)
            for idx,i in enumerate(bests_metrics):
                bests_metrics[i] += [npz.f.arr_1[idx]]
            # Select data to complete the final npz
            bests["generation"] += [generation_number]
            bests["individual"] += [best_weights]
            bests["fitness_train"] += [abs(best_ind)]
            # Add best metrics to bests
            bests.update(bests_metrics)
            bests.update(bests_metrics_test)
            np.savez(os.path.join(folder + file_prefix + '_' + str(generation_number) + '.npz'), **bests)

        return func