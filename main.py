def ML_Classfication(
        df,
        group,
        features,
        decimal_num=3,
        validation_ratio=0.15,
        scoring='roc_auc',
        method='KNeighborsClassifier',
        n_splits=10,
        explain=True,
        shapSet=2,
        explain_numvar=2,
        explain_sample=2,
        searching=False,
        validationCurve=False,
        smooth=False,
        savePath=None,
        dpi=600,
        picFormat='jpeg',
        label='LABEL',
        trainSet=False,
        modelSave=True,
        trainLabel=0,
        randomState=1,
        resultType=0,
        **kwargs,
):
    """
        Input:
            df_input:DataFrame 
            group_name:str 
            validation_ratio:float 
            scoring:str 
            method:str
                        'LogisticRegression':LogisticRegression(**kwargs),
                        'XGBClassifier':XGBClassifier(**kwargs),
                        'RandomForestClassifier':RandomForestClassifier(**kwargs),
                        'SVC':SVC(**kwargs),
                        'KNeighborsClassifier':KNeighborsClassifier(**kwargs),
            n_splits:int 
            explain:bool 
            explain_numvar:int 
            explain_sample:int 
            searching:bool 
            savePath:str 
            **kwargs:dict 

        Return:
            df_dict: dataframe×Öµä£¬°üº¬£º
                    df_train_result: Ä£ÐÍÔÚÑµÁ·Œ¯ÉÏµÄ±íÏÖ
                    df_test_result:  Ä£ÐÍÔÚ²âÊÔŒ¯ÉÏµÄ±íÏÖ
            str_result: ·ÖÎöœá¹û×ÛÊö
            plot_name_list: ÍŒÆ¬ÎÄŒþÃûÁÐ±í
    """
    name_dict = {
        'LogisticRegression': 'logistic',
        'XGBClassifier': 'XGBoost',
        'RandomForestClassifier': 'RandomForest',
        'LGBMClassifier': 'LightGBM',
        'SVC': 'SVM',
        'MLPClassifier': 'MLP',
        'GaussianNB': 'GNB',
        'ComplementNB': 'CNB',
        'AdaBoostClassifier': 'AdaBoost',

        'KNeighborsClassifier': 'KNN',
        'DecisionTreeClassifier': 'DecisionTree',
        'BaggingClassifier': 'Bagging',
    }
    colors = x5.CB91_Grad_BP
    str_time = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    list_name = [group]

    plot_name_dict_save = {}  ##ŽæŽ¢ÍŒÆ¬
    result_model_save = {}  ##Ä£ÐÍŽæŽ¢
    resThreshold = 0  ##ÓÃÓÚŽæŽ¢×îÖÕµÄãÐÖµ
    conf_dic_train, conf_dic_valid, conf_dic_test = {}, {}, {}

    if trainSet:
        df = df[features + [group] + [label]].dropna()
        for fea in features:
            if fea == label or label == group:
                return {'error': '±êÇ©ÁÐ²»ÄÜÔÚËùÔÚÄ£ÐÍÖÐ£¬ÇëÖØÐÂÑ¡ÔñÊýŸÝ»®·Ö±êÇ©ÁÐ£¡'}
    else:
        df = df[features + [group]].dropna()

    binary = True
    u = np.sort(np.unique(np.array(df[group])))
    if len(u) == 2 and set(u) != set([0, 1]):
        y_result = label_binarize(df[group], classes=[ii for ii in u])  # œ«±êÇ©¶þÖµ»¯
        y_result_pd = pd.DataFrame(y_result, columns=[group])
        df = pd.concat([df.drop(group, axis=1), y_result_pd], axis=1)
    elif len(u) > 2:
        if len(u) > 10:
            return {'error': 'ÔÝ²»ÔÊÐíÀà±ðÊýÄ¿ŽóÓÚ10µÄÇé¿ö¡£ÇëŒì²éÒò±äÁ¿È¡ÖµÇé¿ö¡£'}
        binary = False
        if scoring == 'roc_auc':
            scoring = scoring + '_ovo'
        else:
            scoring = scoring + '_macro'
        return {'error': 'ÔÝÊ±Ö»Ö§³Ö¶þ·ÖÀà¡£ÇëŒì²éÒò±äÁ¿È¡ÖµÇé¿ö¡£'}

    if trainSet:
        if isinstance(df[label][0], str):
            trainLabel = str(trainLabel)
        df = df[features + [group] + [label]].dropna()
        train_a = df[df[label] == trainLabel]
        test_a = df[df[label] != trainLabel]
        train_all = train_a.drop(label, axis=1)
        test_all = test_a.drop(label, axis=1)
        # features.remove(fea)
        df = df.drop(label, axis=1)
        Xtrain = train_all.drop(group, axis=1)
        Ytrain = train_all.loc[:, list_name].squeeze(axis=1)
        Xtest = test_all.drop(group, axis=1)
        Ytest = test_all.loc[:, list_name].squeeze(axis=1)
    else:
        df = df[features + [group]].dropna()
        X = df.drop(group, axis=1)
        Y = df.loc[:, list_name].squeeze(axis=1)
        Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=validation_ratio, random_state=randomState, )

    df_dict = {}

    str_result = "²ÉÓÃ%s»úÆ÷Ñ§Ï°·œ·šœøÐÐ·ÖÀà£¬·ÖÀà±äÁ¿Îª%s£¬Ä£ÐÍÖÐµÄ±äÁ¿°üÀš" % (method, group)
    str_result += '¡¢'.join(features)

    if searching == True:
        if method == 'LGBMClassifier':
            searcher = GridSearcherCV('Classification', globals()[method]())
            clf = searcher(Xtrain, Ytrain);
            searcher.report()
        else:
            searcher = RandSearcherCV('Classification', globals()[method]())
            clf = searcher(Xtrain, Ytrain);
            searcher.report()
    elif searching == 'Handle':
        if (method == 'SVC'): kwargs['probability'] = True
        if (method == 'RandomForestClassifier' and kwargs['max_depth'] == 'None'):
            kwargs['max_depth'] = None
        if (method == 'MLPClassifier'):
            hls_vals = str(kwargs['hidden_layer_sizes']).split(',')
            hls_value = ()
            for hls_val in hls_vals:
                try:
                    if int(hls_val) >= 5 and int(hls_val) <= 200:
                        hls_value = hls_value + (int(hls_val),)
                    else:
                        return {'error': 'Çë°ŽÕÕÒªÇóÖØÐÂÉèÖÃÒþ²Ø²ã¿í¶È£¡'}
                except:
                    return {'error': 'ÇëÖØÐÂÉèÉñŸ­ÍøÂçÄ£ÐÍÖÐµÄÒþ²Ø²ã¿í¶È£¡'}
            kwargs['hidden_layer_sizes'] = hls_value
        if (method == 'GaussianNB' and kwargs['priors'] == 'None'):
            kwargs['priors'] = None
        elif (method == 'GaussianNB'):
            pri_vals = str(kwargs['priors']).split(',')
            pri_value = ()
            pri_sum = 0.0
            for pri_val in pri_vals:
                try:
                    pri_sum = float(pri_val) + pri_sum
                    pri_value = pri_value + (float(pri_val),)
                except:
                    return {'error': 'ÇëÖØÐÂÉèÆÓËØ±ŽÒ¶Ë¹Ä£ÐÍÖÐµÄÏÈÑéžÅÂÊ£¡'}
            if len(pri_vals) == len(Y.unique()) and pri_sum == 1.0:
                kwargs['priors'] = pri_value
            else:
                return {'error': 'ÇëÖØÐÂÉèÆÓËØ±ŽÒ¶Ë¹Ä£ÐÍÖÐµÄÏÈÑéžÅÂÊ£¡'}
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)
    elif searching == False:
        # if (method == 'SVC'): kwargs['probability'] = True
        if method == 'SVC':
            kwargs['probability'] = True
        elif method == 'MLPClassifier':
            kwargs['hidden_layer_sizes'] = (20, 10)
            kwargs['max_iter'] = 20
        elif method == 'RandomForestClassifier':
            kwargs['n_estimators'] = 20
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    str_result += "\nÄ£ÐÍ²ÎÊýÎª:\n%s" % dic2str(clf.get_params(), clf.__class__.__name__)
    str_result += "\nÊýŸÝŒ¯Ñù±ŸÊý×ÜŒÆN=%dÀý£¬ÓŠ±äÁ¿ÖÐ°üº¬µÄÀà±ðÐÅÏ¢Îª£º\n" % (df.shape[0])
    group_labels = df[group].unique()
    group_labels.sort()
    for label in group_labels:
        n = sum(df[group] == label)
        str_result += "\t Àà±ð(" + str(label) + ")£ºN=" + str(n) + "Àý\n"

    plot_name_list = x5.plot_learning_curve(
        clf,
        Xtrain,
        Ytrain,
        cv=n_splits,
        scoring=scoring,
        path=savePath,
        dpi=dpi,
        picFormat=picFormat,
    )
    plot_name_dict_save['Ñ§Ï°ÇúÏß'] = plot_name_list[1]
    plot_name_list.pop(len(plot_name_list) - 1)
    ###»­Ð£×ŒÇúÏß
    calibration_curve_name, _ = plot_calibration_curve(clf, Xtrain, Xtest, Ytrain, Ytest, name=method, path=savePath,
                                                       smooth=smooth,
                                                       picFormat=picFormat, dpi=dpi, )
    plot_name_list.append(calibration_curve_name[0])
    plot_name_dict_save['Ð£×ŒÇúÏß'] = calibration_curve_name[1]
    if binary:
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        # »­¶ÔœÇÏß
        plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            lw=1, color='r',
            alpha=0.8,
        )
        plt.grid(which='major', axis='both', linestyle='-.', alpha=0.08, color='grey')

    best_auc = 0.0
    tprs_train, tprs_valid = [], []
    fpr_train_alls, tpr_train_alls = [], []
    mean_fpr = np.linspace(0, 1, 100)
    list_evaluate_dic_train = make_class_metrics_dict()
    list_evaluate_dic_valid = make_class_metrics_dict()
    # KF = KFold(n_splits=n_splits, random_state=randomState,shuffle=True)##StratifiedKFold
    KF = StratifiedKFold(n_splits=n_splits, random_state=randomState, shuffle=True)
    for i, (train_index, valid_index) in enumerate(KF.split(Xtrain, Ytrain)):
        # »®·ÖÑµÁ·Œ¯ºÍÑéÖ€Œ¯
        X_train, X_valid = Xtrain.iloc[train_index], Xtrain.iloc[valid_index]
        Y_train, Y_valid = Ytrain.iloc[train_index], Ytrain.iloc[valid_index]

        # œšÁ¢Ä£ÐÍ(Ä£ÐÍÒÑŸ­¶šÒå)²¢ÑµÁ·
        model = clone(clf).fit(X_train, Y_train)

        # ÀûÓÃclassification_metric_evaluateº¯Êý»ñÈ¡ÔÚÑéÖ€Œ¯µÄÔ€²âÖµ
        fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(model, X_train, Y_train, binary)
        fpr_valid, tpr_valid, metric_dic_valid, _ = classification_metric_evaluate(model, X_valid, Y_valid, binary,
                                                                                   Threshold=metric_dic_train['cutoff'])
        metric_dic_valid.update({'cutoff': metric_dic_train['cutoff']})

        # model selection using validation set
        if metric_dic_valid['AUC'] > best_auc:
            clf = model
            resThreshold = metric_dic_train['cutoff']

        # ŒÆËãËùÓÐÆÀŒÛÖž±ê
        for key in list_evaluate_dic_train.keys():
            list_evaluate_dic_train[key].append(metric_dic_train[key])
            list_evaluate_dic_valid[key].append(metric_dic_valid[key])

        if binary:
            # interp:²åÖµ °Ñœá¹ûÌíŒÓµœtprsÁÐ±íÖÐ
            tprs_valid.append(np.interp(mean_fpr, fpr_valid, tpr_valid))
            tprs_valid[-1][0] = 0.0

            # »­ÍŒ, Ö»ÐèÒªplt.plot(fpr,tpr), ±äÁ¿roc_aucÖ»ÊÇŒÇÂŒaucµÄÖµ, Íš¹ýauc()º¯ÊýŒÆËã³öÀŽ
            if validationCurve:
                plt.plot(
                    fpr_valid, tpr_valid,
                    lw=1, alpha=0.4,
                    label='ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f))' % (
                    i + 1, metric_dic_valid['AUC'], metric_dic_valid['AUC_L'], metric_dic_valid['AUC_U']),
                )

            ##ÑµÁ·Œ¯ROC
            fpr_train_alls.append(fpr_train)
            tpr_train_alls.append(tpr_train)
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_train[-1][0] = 0.0

    if modelSave:
        import pickle
        modelfile = open(savePath + method + str_time + '.pkl', 'wb')
        pickle.dump(clf, modelfile)
        modelfile.close()
        result_model_save['modelFile'] = method + str_time + '.pkl'
        result_model_save['modelFeature'] = features

    if binary:
        mean_tpr_valid = np.mean(tprs_valid, axis=0)
        mean_tpr_valid[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr_valid)  # ŒÆËãÆœŸùAUCÖµ
        aucs_lower, aucs_upper = ci(list_evaluate_dic_valid['AUC'])
        plt.plot(
            mean_fpr, mean_tpr_valid,
            color='b',
            lw=2, alpha=0.8,
            label=r'Mean (validation) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))' % (
            mean_auc, np.mean(list_evaluate_dic_valid['AUC_L']), np.mean(list_evaluate_dic_valid['AUC_U'])),
            # label = r'Mean ROC (auc=%0.3f 0.95CI(%0.3f-%0.3f)' % (mean_auc, aucs_lower, aucs_upper),
        )

    mean_dic_train, stdv_dic_train = {}, {}
    mean_dic_valid, stdv_dic_valid = {}, {}
    for key in list_evaluate_dic_valid.keys():
        mean_dic_train[key] = np.mean(list_evaluate_dic_train[key])
        mean_dic_valid[key] = np.mean(list_evaluate_dic_valid[key])
        if resultType == 0:  ##SD
            stdv_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
            stdv_dic_valid[key] = np.std(list_evaluate_dic_valid[key], axis=0)
        elif resultType == 1:  ##CI
            conf_dic_train[key] = list(ci(list_evaluate_dic_train[key]))
            conf_dic_valid[key] = list(ci(list_evaluate_dic_valid[key]))
    # if resultType == 0:  ##SD
    #    df_train_result = pd.DataFrame([mean_dic_train, stdv_dic_train], index=['Mean', 'SD'])
    #    df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))
    #    df_valid_result = pd.DataFrame([mean_dic_valid, stdv_dic_valid], index=['Mean', 'SD'])
    #    df_valid_result = df_valid_result.applymap(lambda x: round_dec(x, d=decimal_num))

    fpr_test, tpr_test, metric_dic_test, df_test_result = classification_metric_evaluate(clf, Xtest, Ytest, binary,
                                                                                         Threshold=resThreshold)
    metric_dic_test.update({'cutoff': resThreshold})

    # plt.plot(
    #    fpr_test, tpr_test,
    #    lw=1.5, alpha=0.6,
    #    label='Test Set ROC (auc=%0.3f) ' % metric_dic_test['AUC'],
    # )
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Validation ROC')
    plt.legend(loc='lower right', fontsize=5)
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'ROC_curve', 'png', fig, str_time=str_time))
        plot_name_dict_save['ÑéÖ€Œ¯ROCÇúÏß'] = save_fig(savePath, 'ROC_curve', picFormat, fig, str_time=str_time)
    plt.close()

    ##»­ÑµÁ·Œ¯ROC
    if binary:
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        # »­¶ÔœÇÏß
        plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            lw=1, color='r',
            alpha=0.8,
        )
        plt.grid(which='major', axis='both', linestyle='-.', alpha=0.08, color='grey')

        if validationCurve:
            for i in range(len(tpr_train_alls)):
                plt.plot(
                    fpr_train_alls[i], tpr_train_alls[i],
                    lw=1, alpha=0.4,
                    label='ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f)) ' % (
                    i + 1, list_evaluate_dic_train['AUC'][i], list_evaluate_dic_train['AUC_L'][i],
                    list_evaluate_dic_train['AUC_U'][i]),
                )

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_auc_train = auc(mean_fpr, mean_tpr_train)  # ŒÆËãÆœŸùAUCÖµ
        plt.plot(
            mean_fpr, mean_tpr_train,
            color='b',
            lw=1.8, alpha=0.7,
            label=r'Mean (train) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))' % (
            mean_auc_train, np.mean(list_evaluate_dic_train['AUC_L']), np.mean(list_evaluate_dic_train['AUC_U'])),
            # label = r'Mean ROC (auc=%0.3f 0.95CI(%0.3f-%0.3f)' % (mean_auc, aucs_lower, aucs_upper),
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Train ROC')
        plt.legend(loc='lower right', fontsize=5)
        if savePath is not None:
            plot_name_list.append(save_fig(savePath, 'ROC_curve_train', 'png', fig, str_time=str_time))
            plot_name_dict_save['ÑµÁ·Œ¯ROCÇúÏß'] = save_fig(savePath, 'ROC_curve_train', picFormat, fig, str_time=str_time)
        plt.close()

        plot_name_list.reverse()  ###ËùÓÐÍŒÆ¬µ¹ÖÃ

        ###»­²âÊÔŒ¯ROC
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        # »­¶ÔœÇÏß
        plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            lw=1, color='r',
            alpha=0.8,
        )
        plt.grid(which='major', axis='both', linestyle='-.', alpha=0.08, color='grey')
        if smooth:
            from scipy.interpolate import interp1d
            tpr_test_unique, tpr_test_index = np.unique(fpr_test, return_index=True)
            fpr_test_new = np.linspace(min(fpr_test), max(fpr_test), len(fpr_test))
            f = interp1d(tpr_test_unique, tpr_test[tpr_test_index], kind='linear')  ##cubic
            tpr_test_new = f(fpr_test_new)
        else:
            fpr_test_new = fpr_test
            tpr_test_new = tpr_test
        plt.plot(
            fpr_test_new, tpr_test_new,
            lw=1.5, alpha=0.6, color='b',
            label='Test Set ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f)) ' % (
            metric_dic_test['AUC'], metric_dic_test['AUC_L'], metric_dic_test['AUC_U']),
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Test ROC')
        plt.legend(loc='lower right', fontsize=5)
        if savePath is not None:
            plot_name_list.append(save_fig(savePath, 'ROC_curve_test', 'png', fig, str_time=str_time))
            plot_name_dict_save['²âÊÔŒ¯ROCÇúÏß'] = save_fig(savePath, 'ROC_curve_test', picFormat, fig, str_time=str_time)
        plt.close()

    # df_test_result = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))

    if trainSet:
        df_count_c = Xtest.shape[0]
        df_count_r = (Xtest.shape[0] / df.shape[0]) * 100
    else:
        df_count_c = df.shape[0] * validation_ratio
        df_count_r = validation_ratio * 100
    diff, ratio = 0, 0
    if resultType == 1:  ##CI
        str_result += "ÆäÖÐÔÚ×ÜÌåÑù±ŸÖÐËæ»ú³éÈ¡²âÊÔŒ¯N=%dÀý(%3.2f%%)£¬Ê£ÓàÑù±Ÿ×÷ÎªÑµÁ·Œ¯œøÐÐ%dÕÛœ»²æÑéÖ€£¬²¢ÔÚÑéÖ€Œ¯ÖÐµÃµœAUC=%5.4f(%5.4f-%5.4f)¡£\n×îÖÕÄ£ÐÍÔÚ²âÊÔŒ¯ÖÐµÄAUC=%5.4f£¬×ŒÈ·¶È=%5.4f¡£\n" % (
            df_count_c,
            df_count_r,
            n_splits,
            mean_dic_valid['AUC'],
            mean_dic_valid['AUC_L'],
            mean_dic_valid['AUC_U'],
            df_test_result['AUC'].values[0],
            df_test_result['×ŒÈ·¶È'].values[0]
        )
        diff = mean_dic_valid['AUC'] - float(df_test_result.loc['Mean', 'AUC'])
        ratio = diff / float(df_test_result.loc['Mean', 'AUC'])
    elif resultType == 0:  ##SD
        str_result += "ÆäÖÐÔÚ×ÜÌåÑù±ŸÖÐËæ»ú³éÈ¡²âÊÔŒ¯N=%dÀý(%3.2f%%)£¬Ê£ÓàÑù±Ÿ×÷ÎªÑµÁ·Œ¯œøÐÐ%dÕÛœ»²æÑéÖ€£¬²¢ÔÚÑéÖ€Œ¯ÖÐµÃµœAUC=%5.4f¡À%5.4f¡£\n×îÖÕÄ£ÐÍÔÚ²âÊÔŒ¯ÖÐµÄAUC=%5.4f£¬×ŒÈ·¶È=%5.4f¡£\n" % (
            df_count_c,
            df_count_r,
            n_splits,
            mean_dic_valid['AUC'],
            stdv_dic_valid['AUC'],
            df_test_result['AUC'].values[0],
            df_test_result['×ŒÈ·¶È'].values[0]
        )
        diff = float(stdv_dic_valid['AUC']) - float(df_test_result.loc['Mean', 'AUC'])
        ratio = diff / float(df_test_result.loc['Mean', 'AUC'])

    if (not np.isnan(float(diff)) and diff > 0 and (ratio > 0.1)):
        str_result += '×¢ÒâµœAUCÖž±êÏÂÑéÖ€Œ¯±íÏÖ³¬³ö²âÊÔŒ¯{}£¬ÔŒ{}%£¬¿ÉÄÜŽæÔÚ¹ýÄâºÏÏÖÏó¡£œšÒéžü»»Ä£ÐÍ»òÖØÐÂÉèÖÃ²ÎÊý¡£'.format(round(diff, decimal_num),
                                                                                   round(ratio * 100, decimal_num))
    else:
        str_result += 'ŒøÓÚAUCÖž±êÏÂÑéÖ€Œ¯±íÏÖÎŽ³¬³ö²âÊÔŒ¯»ò³¬³ö±ÈÐ¡ÓÚ10%£¬¿ÉÈÏÎªÄâºÏ³É¹Š£¬{}Ä£ÐÍ¿ÉÒÔÓÃÓÚŽËÊýŸÝŒ¯µÄ·ÖÀàœšÄ£ÈÎÎñ¡£'.format(name_dict[method])
    str_result += '\nÈç¹ûÏëœøÒ»²œ¶Ô±Èžü¶à·ÖÀàÄ£ÐÍµÄ±íÏÖ£¬¿ÉÊ¹ÓÃ×ó²àÀžÖÇÄÜ·ÖÎöÖÐµÄ¡®·ÖÀà¶àÄ£ÐÍ×ÛºÏ·ÖÎö¡¯¹ŠÄÜ¡£'

    df_test_result = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))

    if resultType == 1:  ##CI
        for tem in ['AUC', 'AUC_L', 'AUC_U']:
            del conf_dic_train[tem]
            del conf_dic_valid[tem]
        for key in conf_dic_train.keys():
            mean_dic_train[key] = str(round_dec(float(mean_dic_train[key]), d=decimal_num)) + '(' + \
                                  str(round_dec(float(conf_dic_train[key][0]), d=decimal_num)) + '-' + \
                                  str(round_dec(float(conf_dic_train[key][1]), d=decimal_num)) + ')'
            mean_dic_valid[key] = str(round_dec(float(mean_dic_valid[key]), d=decimal_num)) + '(' + \
                                  str(round_dec(float(conf_dic_valid[key][0]), d=decimal_num)) + '-' + \
                                  str(round_dec(float(conf_dic_valid[key][1]), d=decimal_num)) + ')'

        df_train_result = pd.DataFrame([mean_dic_train], index=['Mean'])
        # df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))
        df_valid_result = pd.DataFrame([mean_dic_valid], index=['Mean'])
        df_train_result.iloc[0, 0] = str(round_dec(float(df_train_result.iloc[0, 0]), d=decimal_num)) + '(' + \
                                     str(round_dec(float(df_train_result.iloc[0, -2]), d=decimal_num)) + '-' + \
                                     str(round_dec(float(df_train_result.iloc[0, -1]), d=decimal_num)) + ')'
        df_valid_result.iloc[0, 0] = str(round_dec(float(df_valid_result.iloc[0, 0]), d=decimal_num)) + '(' + \
                                     str(round_dec(float(df_valid_result.iloc[0, -2]), d=decimal_num)) + '-' + \
                                     str(round_dec(float(df_valid_result.iloc[0, -1]), d=decimal_num)) + ')'
        df_train_result.rename(columns={"AUC": 'AUC(95%CI)', 'cutoff': 'cutoff(95%CI)', '×ŒÈ·¶È': '×ŒÈ·¶È(95%CI)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(95%CI)', 'ÌØÒì¶È': 'ÌØÒì¶È(95%CI)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(95%CI)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(95%CI)',
                                        'F1·ÖÊý': 'F1·ÖÊý(95%CI)', 'Kappa': 'Kappa(95%CI)'}, inplace=True)
        df_valid_result.rename(columns={"AUC": 'AUC(95%CI)', 'cutoff': 'cutoff(95%CI)', '×ŒÈ·¶È': '×ŒÈ·¶È(95%CI)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(95%CI)', 'ÌØÒì¶È': 'ÌØÒì¶È(95%CI)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(95%CI)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(95%CI)',
                                        'F1·ÖÊý': 'F1·ÖÊý(95%CI)', 'Kappa': 'Kappa(95%CI)'}, inplace=True)
        df_test_result.iloc[0, 0] = str(df_test_result.iloc[0, 0]) + ' (' + str(df_test_result.iloc[0, -2]) + '-' + str(
            df_test_result.iloc[0, -1]) + ')'
        df_test_result.rename(columns={"AUC": 'AUC (95%CI)'}, inplace=True)
    elif resultType == 0:  ##SD
        for tem in ['AUC_L', 'AUC_U']:
            del stdv_dic_train[tem]
            del stdv_dic_valid[tem]
        for key in stdv_dic_train.keys():
            mean_dic_train[key] = str(round_dec(float(mean_dic_train[key]), d=decimal_num)) + ' (' + \
                                  str(round_dec(float(stdv_dic_train[key]), d=decimal_num)) + ')'
            mean_dic_valid[key] = str(round_dec(float(mean_dic_valid[key]), d=decimal_num)) + ' (' + \
                                  str(round_dec(float(stdv_dic_valid[key]), d=decimal_num)) + ')'

        df_train_result = pd.DataFrame([mean_dic_train], index=['Mean'])
        df_valid_result = pd.DataFrame([mean_dic_valid], index=['Mean'])

        df_train_result.rename(columns={"AUC": 'AUC(SD)', 'cutoff': 'cutoff(SD)', '×ŒÈ·¶È': '×ŒÈ·¶È(SD)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(SD)', 'ÌØÒì¶È': 'ÌØÒì¶È(SD)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(SD)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(SD)',
                                        'F1·ÖÊý': 'F1·ÖÊý(SD)', 'Kappa': 'Kappa(SD)'}, inplace=True)
        df_valid_result.rename(columns={"AUC": 'AUC(SD)', 'cutoff': 'cutoff(SD)', '×ŒÈ·¶È': '×ŒÈ·¶È(SD)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(SD)', 'ÌØÒì¶È': 'ÌØÒì¶È(SD)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(SD)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(SD)',
                                        'F1·ÖÊý': 'F1·ÖÊý(SD)', 'Kappa': 'Kappa(SD)'}, inplace=True)

    df_dictjq = {
        'ÑµÁ·Œ¯œá¹û»ã×Ü': df_train_result.iloc[0:2, 0:8],
        'ÑéÖ€Œ¯œá¹û»ã×Ü': df_valid_result.iloc[0:2, 0:8],
        '²âÊÔŒ¯œá¹û»ã×Ü': df_test_result.iloc[0:2, 0:8],
    }
    df_dict.update(df_dictjq)

    plot_name_dict = {
        'ÑµÁ·Œ¯ROCÇúÏßÍŒ': plot_name_list[0],
        'ÑéÖ€Œ¯ROCÇúÏßÍŒ': plot_name_list[1],
        '²âÊÔŒ¯ROCÇúÏßÍŒ': plot_name_list[4],
        'Ñ§Ï°ÇúÏßÍŒ': plot_name_list[3],
        'Ä£ÐÍÐ£×ŒÇúÏß': plot_name_list[2],
    }

    if binary:  ###»­DCAÇúÏß
        DCA_dict = {}
        prob_pos, p_serie, net_benefit_serie, net_benefit_serie_All = calculate_net_benefit(clf, Xtest, Ytest)
        DCA_dict[name_dict[method]] = {'p_serie': p_serie, 'net_b_s': net_benefit_serie,
                                       'net_b_s_A': net_benefit_serie_All}
        decision_curve_p = plot_decision_curves(DCA_dict, colors=colors, name='Test', savePath=savePath, dpi=dpi,
                                                picFormat=picFormat)
        plot_name_dict['²âÊÔŒ¯DCAÇúÏßÍŒ'] = decision_curve_p[0]
        plot_name_dict_save['²âÊÔŒ¯DCAÇúÏßÍŒ'] = decision_curve_p[1]

    if explain or modelSave:
        import shap
        # from interpret.blackbox import LimeTabular, PartialDependence

        f = lambda x: clf.predict_proba(x)[:, 1]
        med = Xtrain.median().values.reshape((1, Xtrain.shape[1]))

        result_model_save['modelShapValue'] = list(med[0])
        result_model_save['modelName'] = method
        result_model_save['modelClass'] = '»úÆ÷Ñ§Ï°·ÖÀà'
        result_model_save['Threshold'] = resThreshold

    df_shapValue = Xtest
    df_shapValue_show = pd.DataFrame()
    shapValue_list = []
    shapValue_name = []
    if explain:
        if shapSet == 2:  ##Xtrain, Xtest, Ytrain, Ytest
            df_shapValue = Xtest
            if (explain_sample == 4):
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytest)):

                    if (flage1 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and Ytest.iloc[i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_list.append(i)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª1')
                        flage1 = False
                    elif (flage2 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and Ytest.iloc[i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_list.append(i)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª0')
                        flage2 = False
                    elif (flage3 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and Ytest.iloc[i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª1')
                        shapValue_list.append(i)
                        flage3 = False
                    elif (flage4 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and Ytest.iloc[i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª0')
                        shapValue_list.append(i)
                        flage4 = False

                    if (not flage1 and not flage2 and not flage3 and not flage4):
                        break
            else:
                df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0)
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend('shap_Ñù±Ÿ_' + str(i) for i in range(explain_sample))

        elif shapSet == 1:
            df_shapValue = Xtrain
            if (explain_sample == 4):
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytrain)):

                    if (flage1 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and Ytrain.iloc[i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª1')
                        shapValue_list.append(i)
                        flage1 = False
                    elif (flage2 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and Ytrain.iloc[i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª0')
                        shapValue_list.append(i)
                        flage2 = False
                    elif (flage3 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and Ytrain.iloc[i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª1')
                        shapValue_list.append(i)
                        flage3 = False
                    elif (flage4 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and Ytrain.iloc[i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª0')
                        shapValue_list.append(i)
                        flage4 = False

                    if (not flage1 and not flage2 and not flage3 and not flage4):
                        break
            else:
                df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0)
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend('shap_Ñù±Ÿ_' + str(i) for i in range(explain_sample))
        elif shapSet == 0:
            df_shapValue = pd.concat([Xtrain, Xtest], axis=0)
            df_shapValue_Y = pd.concat([Ytrain, Ytest], axis=0)
            if (explain_sample == 4):
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(df_shapValue_Y)):

                    if (flage1 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and df_shapValue_Y.iloc[
                        i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª1')
                        shapValue_list.append(i)
                        flage1 = False
                    elif (flage2 and f(df_shapValue.iloc[i:i + 1, :])[0] >= resThreshold and df_shapValue_Y.iloc[
                        i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª1ÊµŒÊÖµÎª0')
                        shapValue_list.append(i)
                        flage2 = False
                    elif (flage3 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and df_shapValue_Y.iloc[
                        i,] == 1):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª1')
                        shapValue_list.append(i)
                        flage3 = False
                    elif (flage4 and f(df_shapValue.iloc[i:i + 1, :])[0] < resThreshold and df_shapValue_Y.iloc[
                        i,] == 0):
                        df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[i:i + 1, :]], axis=0)
                        shapValue_name.append('shap_Ñù±Ÿ_Ô€²âÖµÎª0ÊµŒÊÖµÎª0')
                        shapValue_list.append(i)
                        flage4 = False

                    if (not flage1 and not flage2 and not flage3 and not flage4):
                        break
            else:
                df_shapValue_show = pd.concat([df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0)
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend('shap_Ñù±Ÿ_' + str(i) for i in range(explain_sample))
        explainer = shap.KernelExplainer(f, med)
        shap_values = explainer.shap_values(df_shapValue)

        if explain_numvar > 0:
            # SHAP beeswarm summary plot
            assert explain_numvar <= len(features)
            fig = shap.summary_plot(shap_values, df_shapValue, show=False)

            if savePath is not None:
                plot_name_dict['SHAP_±äÁ¿¹±Ï×¶È×Üœá'] = save_fig(savePath, 'shap_summary', 'png', fig, str_time=str_time)
                plot_name_dict_save['SHAP_±äÁ¿¹±Ï×¶È×Üœá'] = save_fig(savePath, 'shap_summary', picFormat, fig,
                                                               str_time=str_time)
            plt.close()

            fig1 = shap.summary_plot(shap_values, df_shapValue, plot_type='bar', show=False)

            if savePath is not None:
                plot_name_dict['SHAP_ÖØÒªÐÔÍŒ'] = save_fig(savePath, 'shap_import', 'png', fig1, str_time=str_time)
                plot_name_dict_save['SHAP_ÖØÒªÐÔÍŒ'] = save_fig(savePath, 'shap_import', picFormat, fig1,
                                                            str_time=str_time)
            plt.close()

            # # single feature (Partial Dependence)
            # pdp = PartialDependence(predict_fn=clf.predict_proba, data=Xtrain)
            # pdp_global = pdp.explain_global(name='Partial Dependence')
            # for i in range(explain_numvar):
            #     fig = pdp_global.visualize(key=i)
            #     if savePath is not None:
            #         plot_name_dict['Î¢·ÖÒÀÀµ¶È_±äÁ¿{}'.format(i+1)] = save_fig(savePath, 'partial_dependence_{}'.format(features[i]), '.jpeg', fig)
            #     plt.close()

        if explain_sample > 0:
            assert explain_sample <= len(Ytest)
            # lime = LimeTabular(predict_fn=clf.predict_proba, data=Xtest, random_state=1)
            # lime_local = lime.explain_local(Xtest[:explain_sample], Ytest[:explain_sample], name='LIME')

            for i in range(len(shapValue_list)):
                # SHAP explain
                fig = shap.force_plot(explainer.expected_value, shap_values[shapValue_list[i]],
                                      df_shapValue_show.iloc[i, :], show=False,
                                      figsize=(15, 3), matplotlib=True)
                if savePath is not None:
                    plot_name_dict[shapValue_name[i]] = save_fig(savePath, 'shap_sample_{}'.format(i + 1),
                                                                 'png', fig, str_time=str_time)
                    plot_name_dict_save[shapValue_name[i]] = save_fig(savePath, 'shap_sample_{}'.format(i + 1),
                                                                      picFormat, fig, str_time=str_time)
                plt.close()

                # # LIME explain
                # fig = lime_local.visualize(key=i)
                # if savePath is not None:
                #     plot_name_dict['LIME_Ñù±Ÿ{}'.format(i+1)] = save_fig(savePath, 'lime_{}'.format(i), '.jpeg', fig)
                # plt.close()

    result_dict = {'str_result': {'·ÖÎöœá¹ûÃèÊö': str_result}, 'tables': df_dict,
                   'pics': plot_name_dict, 'save_pics': plot_name_dict_save,
                   'model': result_model_save}
    return result_dict




# -------------------------------------------------------------
# ----------------------·ÖÀà¶àÄ£ÐÍ×ÛºÏ·ÖÎö------------------------
# -------------------------------------------------------------
def two_groups_classfication_multimodels(
        df_input,
        group,
        features,
        methods=[],
        decimal_num=3,
        testsize=0.2,
        boostrap=5,
        randomState=42,
        smooth=False,
        searching=False,
        dpi=600,
        picFormat='jpeg',
        isKFold=True,
        savePath=None,
        resultType=0,
        delong=False,
        **kwargs,
):
    """
        df_input:Dataframe
        features:×Ô±äÁ¿list
        group£ºÒò±äÁ¿str
        testsize: ²âÊÔŒ¯±ÈÀý
        boostrap£ºÖØ²ÉÑùŽÎÊý
        searching:bool ÊÇ·ñœøÐÐ×Ô¶¯Ñ°²Î£¬Ä¬ÈÏÎª·ñ
        savePath:str ÍŒÆ¬ŽæŽ¢Â·Ÿ¶
    """
    str_time = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    dftemp = df_input[features + [group]].dropna()

    features_flag = False
    x = dftemp[features]
    y = dftemp[[group]]

    u = np.sort(np.unique(np.array(dftemp[group])))
    if len(u) == 2 and set(u) != set([0, 1]):
        y_result = label_binarize(dftemp[group], classes=[ii for ii in u])  # œ«±êÇ©¶þÖµ»¯
        y_result_pd = pd.DataFrame(y_result, columns=[group])
        df = pd.concat([dftemp.drop(group, axis=1), y_result_pd], axis=1)
        x = df[features]
        y = df[[group]]
    elif len(u) > 2:
        return {'error': 'ÔÝÊ±Ö»Ö§³Ö¶þ·ÖÀà¡£ÇëŒì²éÒò±äÁ¿È¡ÖµÇé¿ö¡£'}

    name_dict = {
        'LogisticRegression': 'logistic',
        'XGBClassifier': 'XGBoost',
        'RandomForestClassifier': 'RandomForest',
        'LGBMClassifier': 'LightGBM',
        'SVC': 'SVM',
        'MLPClassifier': 'MLP',
        'GaussianNB': 'GNB',
        'ComplementNB': 'CNB',
        'AdaBoostClassifier': 'AdaBoost',

        'KNeighborsClassifier': 'KNN',
        'DecisionTreeClassifier': 'DecisionTree',
        'BaggingClassifier': 'Bagging',
    }
    if len(methods) == 0:
        methods = [
            'LogisticRegression',
            'XGBClassifier',
            'RandomForestClassifier',
            # 'SVC',
            # 'MLPClassifier',
            # 'AdaBoostClassifier',
            # 'KNeighborsClassifier',
            # 'DecisionTreeClassifier',
            # 'BaggingClassifier',
        ]
    str_result = 'ÒÑ²ÉÓÃ¶àÖÖ»úÆ÷Ñ§Ï°Ä£ÐÍ³¢ÊÔÍê³ÉÊýŸÝÑù±Ÿ·ÖÀàÈÎÎñ£¬°üÀš£º{}¡£ž÷Ä£ÐÍµÄ²ÎÊýÖµÑ¡È¡Çé¿öÈçÏÂËùÊŸ£º\n\n'.format(methods)

    plot_name_list = []
    plot_name_dict_save = {}
    plot_name_dict ={}

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    # »­¶ÔœÇÏß
    ax.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=1, color='r',
        alpha=1.0,
    )
    ax.grid(which='major', axis='both', linestyle='-.', alpha=0.3, color='grey')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.tick_params(top=False, right=False)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Validation ROC Curve')

    mean_fpr = np.linspace(0, 1, 100)
    colors = x5.CB91_Grad_BP

    df_0 = pd.DataFrame(columns=list(make_class_metrics_dict().keys()), index=[0])
    df_0_test = df_0.copy()

    df_plot = pd.DataFrame(columns=['method', 'mean', 'std'])

    fpr_train_alls, tpr_train_alls, train_method_alls, mean_auc_train_alls = [], [], [], []
    fraction_of_positives_alls, mean_predicted_value_alls, clf_score_alls = [], [], []
    AUC_95CI_test, AUC_95CI_SD_test, AUC_95CI_train, AUC_95CI_SD_train = [], [], [], []

    DCA_dict = {}
    model_test_data_all = {}

    X_train_ps, Y_train_ps, model_train_s = [], [], []
    X_test_ps, Y_test_ps = [], []
    name = []

    for i, method in enumerate(methods):
        tprs_train, tprs_test = [], []

        name.append(name_dict[method])
        if searching == True:
            if method == 'LGBMClassifier':
                searcher = GridSearcherCV('Classification', globals()[method]())
                selected_model = searcher(x, y)
            else:
                searcher = RandSearcherCV('Classification', globals()[method]())
                selected_model = searcher(x, y)  # ; searcher.report()
        elif searching == False:
            # selected_model = globals()[method]() if (method != 'SVC') else globals()[method](probability=True)
            if method == 'SVC':
                selected_model = globals()[method](probability=True)
            elif method == 'MLPClassifier':
                selected_model = globals()[method](hidden_layer_sizes=(20, 10), max_iter=20)
            elif method == 'RandomForestClassifier':
                selected_model = globals()[method](n_estimators=20)
            else:
                selected_model = globals()[method]()
        elif searching == 'Handle':
            method_dicts = kwargs
            if i == 0:
                me_count = True
                for me_list in methods:
                    if me_list in method_dicts.keys():
                        me_count = False
                        continue
                if me_count:
                    return {'error': 'ÇëÉèÖÃÒªµ÷²ÎµÄÄ£ÐÍ£¡'}
            if method in method_dicts.keys():
                method_dict = {}
                if (method == 'SVC'):
                    method_dict.update({'probability': True})
                method_dict.update(method_dicts[method])
                if (method == 'RandomForestClassifier' and method_dict['max_depth'] == 'None'):
                    method_dict['max_depth'] = None
                if (method == 'MLPClassifier'):
                    hls_vals = str(method_dict['hidden_layer_sizes']).split(',')
                    hls_value = ()
                    for hls_val in hls_vals:
                        try:
                            if int(hls_val) >= 5 and int(hls_val) <= 200:
                                hls_value = hls_value + (int(hls_val),)
                            else:
                                return {'error': 'Çë°ŽÕÕÒªÇóÖØÐÂÉèÖÃÒþ²Ø²ã¿í¶È£¡'}
                        except:
                            return {'error': 'ÇëÖØÐÂÉèÉñŸ­ÍøÂçÄ£ÐÍÖÐµÄÒþ²Ø²ã¿í¶È£¡'}
                    method_dict['hidden_layer_sizes'] = hls_value
                if (method == 'GaussianNB' and method_dict['priors'] == 'None'):
                    method_dict['priors'] = None
                elif (method == 'GaussianNB'):
                    pri_vals = str(method_dict['priors']).split(',')
                    pri_value = ()
                    pri_sum = 0.0
                    for pri_val in pri_vals:
                        try:
                            pri_sum = float(pri_val) + pri_sum
                            pri_value = pri_value + (float(pri_val),)
                        except:
                            return {'error': 'ÇëÖØÐÂÉèÆÓËØ±ŽÒ¶Ë¹Ä£ÐÍÖÐµÄÏÈÑéžÅÂÊ£¡'}
                    if len(pri_vals) == len(y.unique()) and pri_sum == 1.0:
                        method_dict['priors'] = pri_value
                    else:
                        return {'error': 'ÇëÖØÐÂÉèÆÓËØ±ŽÒ¶Ë¹Ä£ÐÍÖÐµÄÏÈÑéžÅÂÊ£¡'}
                selected_model = globals()[method](**method_dict)
            else:
                if method == 'LGBMClassifier':
                    searcher = GridSearcherCV('Classification', globals()[method]())
                    selected_model = searcher(x, y)
                else:
                    searcher = RandSearcherCV('Classification', globals()[method]())
                    selected_model = searcher(x, y)  # ; searcher.report()

        list_evaluate_dic_train = make_class_metrics_dict()
        list_evaluate_dic_test = make_class_metrics_dict()

        clf_score = 1
        fraction_of_positives = np.array([1])
        mean_predicted_value = np.array([1])

        p_serie_s_te, net_benefit_serie_s_te, net_benefit_serie_All_s_te = [], [], []
        data_all = {}
        test_data_delong = {}
        conf_dic_train, conf_dic_test = {}, {}
        if isKFold:
            # KF = KFold(n_splits=boostrap, random_state=42,shuffle=True)
            KF = StratifiedKFold(n_splits=boostrap, random_state=randomState, shuffle=True)
            for i_k, (train_index, valid_index) in enumerate(KF.split(x, y)):
                # »®·ÖÑµÁ·Œ¯ºÍÑéÖ€Œ¯
                Xtrain, Xtest = x.iloc[train_index], x.iloc[valid_index]
                Ytrain, Ytest = y.iloc[train_index], y.iloc[valid_index]
                data_all.update({i_k: {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Xtest': Xtest, 'Ytest': Ytest}})
                test_data_delong.update({i_k: np.array(Ytest).T[0]})
        else:
            for index in range(0, boostrap):
                if searching == 'Handle':
                    Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=testsize, random_state=index)
                else:
                    Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=testsize)
                data_all.update({index: {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Xtest': Xtest, 'Ytest': Ytest}})
                test_data_delong.update({index: np.array(Ytest).T[0]})
        if method == methods[0]:
            model_test_data_all.update({'original': test_data_delong})
        # for index in range(0, boostrap):
        test_all_data_delong = {}

        X_train_p, Y_train_p, model_train = [], [], []
        X_test_p, Y_test_p = [], []

        for data_key, data_value in data_all.items():
            Xtrain, Ytrain, Xtest, Ytest = data_value['Xtrain'], data_value['Ytrain'], data_value['Xtest'], data_value[
                'Ytest']

            model = clone(selected_model).fit(Xtrain, Ytrain)
            ####################################
            # if data_key == 0:
            X_train_p.append(Xtrain)
            Y_train_p.append(Ytrain)
            model_train.append(model)
            X_test_p.append(Xtest)
            Y_test_p.append(Ytest)

            ##########################################
            Yprob = model.predict_proba(Xtest)[:, 1]
            test_all_data_delong.update({data_key: Yprob})
            prob_pos, p_serie, net_benefit_serie, net_benefit_serie_All = calculate_net_benefit(model, Xtest, Ytest)
            p_serie_s_te.append(p_serie)
            net_benefit_serie_s_te.append(net_benefit_serie)
            net_benefit_serie_All_s_te.append(net_benefit_serie_All)
            """
            if hasattr(model, "predict_proba"):
                prob_pos = model.predict_proba(Xtest)[:, 1]
            else:  # use decision function
                prob_pos = model.decision_function(Xtest)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            """
            clf_score1 = brier_score_loss(Ytest, prob_pos, pos_label=y[group].max())  ##strategy='quantile',
            if clf_score > clf_score1:
                clf_score = clf_score1
                fraction_of_positives, mean_predicted_value = calibration_curve(Ytest, prob_pos,
                                                                                n_bins=10)

            # ÀûÓÃclassification_metric_evaluateº¯Êý»ñÈ¡ÔÚ²âÊÔŒ¯µÄÔ€²âÖµ
            try:
                fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(model, Xtrain, Ytrain)

                fpr_test, tpr_test, metric_dic_test, _ = classification_metric_evaluate(model, Xtest, Ytest,
                                                                                        Threshold=metric_dic_train[
                                                                                            'cutoff'])
                metric_dic_test.update({'cutoff': metric_dic_train['cutoff']})
            except Exception as e:
                return {'error': 'ÊýŸÝ²»Ÿùºâ£¬ÖÁÉÙÓÐÒ»×éÑéÖ€Œ¯ÖÐŽæÔÚœáŸÖÈ«²¿Îª0»òÕß1µÄÊýŸÝ£¡ÇëÑ¡ÔñÁíÍâÒ»ÖÖ·œ·šÖØ²ÉÑù£šœ»²æÑéÖ€£©µÄ·œ·šŽŠÀí£¡'}

            # interp:²åÖµ °Ñœá¹ûÌíŒÓµœtprsÁÐ±íÖÐ
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
            tprs_train[-1][0] = 0.0
            tprs_test[-1][0] = 0.0

            # ŒÆËãËùÓÐÆÀŒÛÖž±ê
            for key in list_evaluate_dic_train.keys():
                list_evaluate_dic_train[key].append(metric_dic_train[key])
                list_evaluate_dic_test[key].append(metric_dic_test[key])

        model_test_data_all.update({method: test_all_data_delong})
        DCA_dict[name_dict[method]] = {'p_serie': p_serie_s_te, 'net_b_s': net_benefit_serie_s_te,
                                       'net_b_s_A': net_benefit_serie_All_s_te}

        X_train_ps.append(X_train_p)
        Y_train_ps.append(Y_train_p)
        model_train_s.append(model_train)

        X_test_ps.append(X_test_p)
        Y_test_ps.append(Y_test_p)

        ###»­Ð£×ŒÇúÏß
        # X_train, X_test, Y_train, Y_test = TTS(x, y, test_size=testsize, random_state=0)
        # model_CC = clone(selected_model).fit(X_train, Y_train)
        # y_pred = model.predict(Xtest)
        # if hasattr(model_CC, "predict_proba"):
        #    prob_pos = model_CC.predict_proba(X_test)[:, 1]
        # else:  # use decision function
        #    prob_pos = model_CC.decision_function(X_test)
        #    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        # clf_score = brier_score_loss(Y_test, prob_pos, pos_label=y.max())##strategy='quantile',
        # fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos,
        #                                                                        n_bins=10)
        clf_score_alls.append(clf_score)
        fraction_of_positives_alls.append(fraction_of_positives)
        mean_predicted_value_alls.append(mean_predicted_value)

        for key in list_evaluate_dic_train.keys():

            metric_dic_train[key] = np.mean(list_evaluate_dic_train[key])
            metric_dic_test[key] = np.mean(list_evaluate_dic_test[key])

            if resultType == 0:  ##SD
                list_evaluate_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
                list_evaluate_dic_test[key] = np.std(list_evaluate_dic_test[key], axis=0)
            elif resultType == 1:  ##CI
                conf_dic_train[key] = list(ci(list_evaluate_dic_train[key]))
                conf_dic_test[key] = list(ci(list_evaluate_dic_test[key]))
        result_dic_train = metric_dic_train
        result_dic_test = metric_dic_test
        if resultType == 0:  ##SD
            for tem in ['AUC_L', 'AUC_U']:
                del list_evaluate_dic_train[tem]
                del list_evaluate_dic_test[tem]
            for key in list_evaluate_dic_train.keys():
                if key == 'AUC':
                    result_dic_train['AUC(95%CI)'] = str(round_dec(float(metric_dic_train[key]), d=decimal_num)) + '(' + \
                                                     str(round_dec(float(list_evaluate_dic_train[key]),
                                                                   d=decimal_num)) + ')'

                    result_dic_test['AUC(95%CI)'] = str(round_dec(float(metric_dic_test[key]), d=decimal_num)) + '(' + \
                                                    str(round_dec(float(list_evaluate_dic_test[key]),
                                                                  d=decimal_num)) + ')'
                else:
                    result_dic_train[key] = str(round_dec(float(metric_dic_train[key]), d=decimal_num)) + '(' + \
                                            str(round_dec(float(list_evaluate_dic_train[key]), d=decimal_num)) + ')'
                    result_dic_test[key] = str(round_dec(float(metric_dic_test[key]), d=decimal_num)) + '(' + \
                                           str(round_dec(float(list_evaluate_dic_test[key]), d=decimal_num)) + ')'
        elif resultType == 1:
            for tem in ['AUC_L', 'AUC_U']:
                del conf_dic_train[tem]
                del conf_dic_test[tem]
            for key in conf_dic_train.keys():
                if key == 'AUC':
                    result_dic_train['AUC(95%CI)'] = str(round_dec(float(metric_dic_train[key]), decimal_num)) + ' (' + \
                                                     str(round_dec(float(metric_dic_train['AUC_L']),
                                                                   decimal_num)) + '-' + \
                                                     str(round_dec(float(metric_dic_train['AUC_U']), decimal_num)) + '£©'

                    result_dic_test['AUC(95%CI)'] = str(round_dec(float(metric_dic_test[key]), decimal_num)) + ' (' + \
                                                    str(round_dec(float(metric_dic_test['AUC_L']), decimal_num)) + '-' + \
                                                    str(round_dec(float(metric_dic_test['AUC_U']), decimal_num)) + '£©'
                else:
                    result_dic_train[key] = str(round_dec(float(metric_dic_train[key]), d=decimal_num)) + '(' + \
                                            str(round_dec(float(conf_dic_train[key][0]), d=decimal_num)) + '-' + \
                                            str(round_dec(float(conf_dic_train[key][1]), d=decimal_num)) + ')'
                    result_dic_test[key] = str(round_dec(float(metric_dic_test[key]), d=decimal_num)) + '(' + \
                                           str(round_dec(float(conf_dic_test[key][0]), d=decimal_num)) + '-' + \
                                           str(round_dec(float(conf_dic_test[key][1]), d=decimal_num)) + ')'
        df_train_result = pd.DataFrame([result_dic_train], index=['Mean'])
        df_test_result = pd.DataFrame([result_dic_test], index=['Mean'])
        df_train_result['·ÖÀàÄ£ÐÍ'] = name_dict[method]
        df_test_result['·ÖÀàÄ£ÐÍ'] = name_dict[method]

        AUC_95CI_test.append(list(df_test_result.iloc[0, -4:-2]))
        AUC_95CI_train.append(list(df_train_result.iloc[0, -4:-2]))

        df_0 = pd.concat([df_0, df_train_result])
        df_0_test = pd.concat([df_0_test, df_test_result])

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_test = np.mean(tprs_test, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_tpr_test[-1] = 1.0
        mean_auc_train = auc(mean_fpr, mean_tpr_train)  # ŒÆËãÑµÁ·Œ¯ÆœŸùAUCÖµ
        mean_auc_test = auc(mean_fpr, mean_tpr_test)

        ###»­ÑµÁ·Œ¯ROC
        fpr_train_alls.append(mean_fpr)
        tpr_train_alls.append(mean_tpr_train)
        train_method_alls.append(method)
        mean_auc_train_alls.append(mean_auc_train)

        std_value = 0
        if resultType == 0:
            std_value = list_evaluate_dic_test['AUC']
        elif resultType == 1:
            std_value = (conf_dic_train['AUC'][1] - conf_dic_train['AUC'][0]) / 2
        df_plot = df_plot.append({
            'method': name_dict[method],
            'mean': mean_auc_test,
            'std': std_value,
        }, ignore_index=True)
        ax.plot(mean_fpr, mean_tpr_test, c=colors[i], label=name_dict[method] + '(AUC = %0.3f 95%%CI (%0.3f-%0.3f))' % (
            df_test_result.iloc[0, 0], df_test_result.iloc[0, -4], df_test_result.iloc[0, -3]),
                lw=1.5, alpha=1)
        str_result += method + ': AUC=' + str(round_dec(mean_auc_train, decimal_num)) + ';  Ä£ÐÍ²ÎÊý:\n' + dic2str(
            selected_model.get_params(),
            method) + '\n'
    ###Ä£ÐÍµÂÁúŒì²â
    if delong:
        delong_z, delong_p = [], []
        for i in range(boostrap):
            zzz, ppp = [], []
            for method1 in methods:
                zz, pp = [], []
                for method2 in methods:
                    z, p = delong_roc_test(model_test_data_all['original'][i], model_test_data_all[method1][i],
                                           model_test_data_all[method2][i])
                    zz.append(z[0][0])
                    pp.append(p[0][0])
                zzz.append(zz)
                ppp.append(pp)
            delong_z.append(zzz)
            delong_p.append(ppp)
        if boostrap == 1:
            delong_zz1 = pd.DataFrame(reduce(lambda x, y: np.array(x) + np.array(y), delong_z),
                                      index=methods,
                                      columns=methods)
            delong_pp1 = pd.DataFrame(reduce(lambda x, y: np.array(x) + np.array(y), delong_p),
                                      index=methods,
                                      columns=methods)
        else:
            delong_zz1 = pd.DataFrame(reduce(lambda x, y: np.array(x) + np.array(y), delong_z) / len(delong_z),
                                      index=methods,
                                      columns=methods)
            delong_pp1 = pd.DataFrame(reduce(lambda x, y: np.array(x) + np.array(y), delong_p) / len(delong_p),
                                      index=methods,
                                      columns=methods)

        delong_zz = delong_zz1.applymap(lambda x: round_dec(x, d=decimal_num))
        delong_pp = delong_pp1.applymap(lambda x: round_dec(x, d=decimal_num))

    # ymin = min([y - dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    # ymax = max([y + dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    # ymin, ymax = ymin - (ymax - ymin) / 4.0, ymax + (ymax - ymin) / 10.0
    if boostrap != 1:
        ymax = np.max(df_plot['mean']) + np.max(df_plot['std']) + (np.max(df_plot['mean']) - np.min(df_plot['mean'])) / 4
        ymin = np.min(df_plot['mean']) - np.max(df_plot['std']) - (np.max(df_plot['mean']) - np.min(df_plot['mean'])) / 4

        ymax = math.ceil(ymax * 100) / 100
        ymin = int(ymin * 100) / 100

    ax.legend(loc="lower right", fontsize=5)
    ax.legend(loc="lower right", fontsize=5)

    df_test_auc = []
    if savePath is not None:
        plot_name_list.append(save_fig(savePath, 'valid_ROC_curve', 'png', fig, str_time=str_time))
        plot_name_dict_save['ÑéÖ€Œ¯ROCÇúÏß'] = save_fig(savePath, 'valid_ROC_curve', picFormat, fig, str_time=str_time)

        # »­ÑµÁ·Œ¯ROC
        fig1 = plt.figure(figsize=(4, 4), dpi=dpi)
        # »­¶ÔœÇÏß
        plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            lw=1, color='r',
            alpha=0.8,
        )
        plt.grid(which='major', axis='both', linestyle='-.', alpha=0.3, color='grey')

        for i in range(len(fpr_train_alls)):
            df_test_auc.append(df_0.iloc[i + 1]['AUC'])
            plt.plot(
                fpr_train_alls[i], tpr_train_alls[i],
                lw=1.5, alpha=0.9,
                c=colors[i],
                label=name_dict[train_method_alls[i]] + '(AUC = %0.3f 95%%CI (%0.3f-%0.3f))' % (
                    df_0.iloc[i + 1]['AUC'], AUC_95CI_train[i][0], AUC_95CI_train[i][1])
            )

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Train ROC Curve')
        plt.legend(loc='lower right', fontsize=5)

        plot_name_list.append(save_fig(savePath, 'ROC_Train_curve', 'png', fig1, str_time=str_time))
        plot_name_dict_save['ÑµÁ·Œ¯ROCÇúÏßÍŒ'] = save_fig(savePath, 'ROC_Train_curve', picFormat, fig1, str_time=str_time)
        plot_name_list.reverse()  ###ËùÓÐÍŒÆ¬µ¹ÖÃ

        if boostrap != 1:
            # df_plot.drop('mean', axis=1)
            # df_plot.loc[:,'mean']=pd.Series(df_test_auc,name='mean')
            plot_name_list += x5.forest_plot(
                df_input=df_plot,
                name='method', value='mean', err='std', direct='horizontal',
                fig_size=[len(methods) + 3, 9],
                ylim=[ymin, ymax],
                title='Forest Plot of Each Model AUC Score ',
                path=savePath,
                dpi=dpi,
                picFormat=picFormat,
            )
            plot_name_dict_save['ÑéÖ€Œ¯¶àÄ£ÐÍÉ­ÁÖÍŒ'] = plot_name_list[len(plot_name_list) - 1]
            plot_name_list.pop(len(plot_name_list) - 1)
    plt.close()
    ###»­Ð£×ŒÇúÏß
    if savePath is not None:
        from scipy.optimize import curve_fit
        from scipy.interpolate import make_interp_spline
        def fit_f(x, a, b):
            return a * np.arcsin(x) + b

        def fit_show(x, y_fit):
            a, b = y_fit.tolist()
            return a * np.arcsin(x) + b

        fig, ax1 = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        for i in range(len(mean_predicted_value_alls)):
            if smooth and len(fraction_of_positives_alls[i]) >= 3:
                x_new = np.linspace(min(mean_predicted_value_alls[i]), max(mean_predicted_value_alls[i]),
                                    len(fraction_of_positives_alls[i]) * 10)
                try:
                    p_fit, _ = curve_fit(fit_f, mean_predicted_value_alls[i], fraction_of_positives_alls[i],
                                         maxfev=10000)
                    y_smooth = fit_show(x_new, p_fit)
                    # y_fit = np.polyfit(mean_predicted_value_alls[i], fraction_of_positives_alls[i], 3)
                    # y_smooth = f_fit(x_new, y_fit)
                    # y_smooth = spline(mean_predicted_value_alls[i], fraction_of_positives_alls[i], x_new)
                    ax1.plot(x_new, y_smooth, c=colors[i],
                             label="%s (%1.3f)" % (name_dict[methods[i]], clf_score_alls[i]))
                except Exception as e:
                    ax1.plot(mean_predicted_value_alls[i], fraction_of_positives_alls[i], "s-", c=colors[i],
                             label="%s (%1.3f)" % (name_dict[methods[i]], clf_score_alls[i]))
            else:

                ax1.plot(mean_predicted_value_alls[i], fraction_of_positives_alls[i], "s-", c=colors[i],
                         label="%s (%1.3f)" % (name_dict[methods[i]], clf_score_alls[i]))

        ax1.set_xlabel("Mean predicted value")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        plt.gca()
        plt.close()
        plot_name = "Calibration_curve_" + str_time
        plot_name_list.append(save_fig(savePath, plot_name, 'png', fig, str_time=str_time))
        plot_name_dict_save['ÑéÖ€Œ¯¶àÄ£ÐÍÐ£×ŒÇúÏß'] = save_fig(savePath, plot_name, picFormat, fig, str_time=str_time)

    ###»­DCAÇúÏß
    if savePath is not None:
        decision_curve_p = plot_decision_curves(DCA_dict, colors=colors, name='Valid', savePath=savePath, dpi=dpi,
                                                picFormat=picFormat)
        plot_name_list.append(decision_curve_p[0])
        plot_name_dict_save['ÑéÖ€Œ¯DCAÇúÏßÍŒ'] = decision_curve_p[1]
    ###»­PRÇúÏß
    if savePath is not None:
        # from sklearn.metrics import plot_precision_recall_curve
        from AnalysisFunction.X_5_SmartPlot import plot_precision_recall_curve
        fig = plot_precision_recall_curve(model_train_s, X_train_ps, Y_train_ps, name=name, picname='train')
        plot_name_dict['ÑµÁ·Œ¯¶àÄ£ÐÍPRÇúÏß'] = save_fig(savePath, 'PR_train', 'png', fig, str_time=str_time)
        plot_name_dict_save['ÑµÁ·Œ¯¶àÄ£ÐÍPRÇúÏß'] = save_fig(savePath, 'PR_train', picFormat, fig, str_time=str_time)
        plt.close(fig)
        fig =plot_precision_recall_curve(model_train_s, X_test_ps, Y_test_ps, name=name, picname='Validation')
        plot_name_dict['ÑéÖ€Œ¯¶àÄ£ÐÍPRÇúÏß'] = save_fig(savePath, 'PR_valid', 'png', fig, str_time=str_time)
        plot_name_dict_save['ÑéÖ€Œ¯¶àÄ£ÐÍPRÇúÏß'] = save_fig(savePath, 'PR_vlid', picFormat, fig, str_time=str_time)
        plt.close(fig)

    df_train_result1 = df_0.drop([0])
    df_test_result1 = df_0_test.drop([0])

    classfier = df_train_result1.pop('·ÖÀàÄ£ÐÍ')

    df_train_result = df_train_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_train_result.insert(0, '·ÖÀàÄ£ÐÍ', classfier)

    df_test_result1.pop('·ÖÀàÄ£ÐÍ')
    df_test_result = df_test_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_test_result.insert(0, '·ÖÀàÄ£ÐÍ', classfier)

    AUC_95CI_tr = df_train_result.pop('AUC(95%CI)')
    df_train_result.insert(1, 'AUC(95%CI)', AUC_95CI_tr)
    AUC_95CI_te = df_test_result.pop('AUC(95%CI)')
    df_test_result.insert(1, 'AUC(95%CI)', AUC_95CI_te)

    df_train_result = df_train_result.drop(['AUC_L', 'AUC_U'], axis=1)
    df_test_result = df_test_result.drop(['AUC_L', 'AUC_U'], axis=1)

    if features_flag:
        df_count_r = round_dec(Xtest.shape[0] / x.shape[0], decimal_num)
    else:
        df_count_r = round_dec(testsize, decimal_num)
    if isKFold:
        str_result += '\nÏÂÊŸÉ­ÁÖÍŒÕ¹ÊŸÁËž÷Ä£ÐÍœøÐÐ' + group + 'Ô€²âµÄROCœá¹û,ÍŒÖÐµÄÎó²îÏßÎªROCŸùÖµŒ°SD¡£\n' \
                      + 'Ä£ÐÍµÄROCŸùÖµŒ°SDµÄÊÇÍš¹ý' + str(boostrap) + 'ÕÛœ»²æÑéÖ€,' + 'Ä£ÐÍÖÐµÄ±äÁ¿°üÀš' \
                      + ','.join(features) + '¡£\n'
    else:
        str_result += '\nÏÂÊŸÉ­ÁÖÍŒÕ¹ÊŸÁËž÷Ä£ÐÍœøÐÐ' + group + 'Ô€²âµÄROCœá¹û,ÍŒÖÐµÄÎó²îÏßÎªROCŸùÖµŒ°SD¡£\n' \
                      + 'Ä£ÐÍµÄROCŸùÖµŒ°SDµÄÊÇÍš¹ý¶àŽÎÖØžŽ²ÉÑùŒÆËã£¬ÖØžŽ²ÉÑùŽÎÊýÎª' + str(boostrap) + 'ŽÎ,' \
                      + 'Ã¿Ò»ŽÎÖØ²ÉÑùÑµÁ·µÄÑéÖ€Œ¯ÕŒ×ÜÌåÑù±ŸµÄ' + str(df_count_r * 100) + '%,ÑµÁ·Œ¯ÕŒ' \
                      + str((1 - df_count_r) * 100) + '%,' + 'Ä£ÐÍÖÐµÄ±äÁ¿°üÀš' \
                      + ','.join(features) + '¡£\n'

    best_ = df_train_result.loc[df_train_result.index == 'Mean'].sort_values(by='AUC', ascending=False).head(1)
    name_train = best_.iloc[0]['·ÖÀàÄ£ÐÍ']
    str_result += 'ÔÚÄ¿Ç°ËùÓÐÄ£ÐÍÖÐ£¬ÑµÁ·Œ¯±íÏÖ×îŒÑÕßÎª{}£šÒÀŸÝAUCÅÅÐò£©£¬ÔÚž÷ÆÀŒÛ±ê×ŒÖÐÆäÔÚÑµÁ·Œ¯¶ÔÓŠ·ÖÊý·Ö±ðÎª£º\n'.format(name_train)
    for col in best_.columns[1:]:
        str_result += '\t{}£º{}\n'.format(col, best_.iloc[0][col])

    best_ = df_test_result.loc[df_test_result.index == 'Mean'].sort_values(by='AUC', ascending=False).head(1)
    name_test = best_.iloc[0]['·ÖÀàÄ£ÐÍ']
    str_result += 'ÑéÖ€Œ¯±íÏÖ×îŒÑÕßÎª{}£šÒÀŸÝAUCÅÅÐò£©£¬ÔÚž÷ÆÀŒÛ±ê×ŒÖÐÆäÔÚÑéÖ€Œ¯¶ÔÓŠ·ÖÊý·Ö±ðÎª£º\n'.format(name_test)
    for col in best_.columns[1:]:
        str_result += '\t{}£º{}\n'.format(col, best_.iloc[0][col])

    if (name_test == name_train):
        str_result += '¶þÕßÎÇºÏ£¬¿ÉÒÔÈÏÎª{}ÊÇÕë¶ÔŽËÊýŸÝŒ¯µÄ×îŒÑÄ£ÐÍÑ¡Ôñ¡£'.format(name_train)
    else:
        str_result += '¶þÕß²»ÎÇºÏ£¬{}Œ«¿ÉÄÜŽæÔÚ¹ýÄâºÏÏÖÏó£¬{}¿ÉÄÜÎÈ¶šÐÔÏà¶ÔœÏºÃ¡£ŸßÌåÄ£ÐÍÑ¡Ôñ¿ÉžùŸÝÏÂ±íÏêÏžÆÀ·ÖÐÅÏ¢œøÐÐÈ¡Éá¡£'.format(name_train, name_test)

    if resultType == 0:
        df_train_result.rename(columns={"AUC(95%CI)": 'AUC(SD)', 'cutoff': 'cutoff(SD)', '×ŒÈ·¶È': '×ŒÈ·¶È(SD)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(SD)', 'ÌØÒì¶È': 'ÌØÒì¶È(SD)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(SD)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(SD)',
                                        'F1·ÖÊý': 'F1·ÖÊý(SD)', 'Kappa': 'Kappa(SD)'}, inplace=True)
        df_test_result.rename(columns={"AUC(95%CI)": 'AUC(SD)', 'cutoff': 'cutoff(SD)', '×ŒÈ·¶È': '×ŒÈ·¶È(SD)',
                                       'ÁéÃô¶È': 'ÁéÃô¶È(SD)', 'ÌØÒì¶È': 'ÌØÒì¶È(SD)',
                                       'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(SD)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(SD)',
                                       'F1·ÖÊý': 'F1·ÖÊý(SD)', 'Kappa': 'Kappa(SD)'}, inplace=True)
    elif resultType == 1:
        df_train_result.rename(columns={'cutoff': 'cutoff(95%CI)', '×ŒÈ·¶È': '×ŒÈ·¶È(95%CI)',
                                        'ÁéÃô¶È': 'ÁéÃô¶È(95%CI)', 'ÌØÒì¶È': 'ÌØÒì¶È(95%CI)',
                                        'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(95%CI)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(95%CI)',
                                        'F1·ÖÊý': 'F1·ÖÊý(95%CI)', 'Kappa': 'Kappa(95%CI)'}, inplace=True)
        df_test_result.rename(columns={'cutoff': 'cutoff(95%CI)', '×ŒÈ·¶È': '×ŒÈ·¶È(95%CI)',
                                       'ÁéÃô¶È': 'ÁéÃô¶È(95%CI)', 'ÌØÒì¶È': 'ÌØÒì¶È(95%CI)',
                                       'ÑôÐÔÔ€²âÖµ': 'ÑôÐÔÔ€²âÖµ(95%CI)', 'ÒõÐÔÔ€²âÖµ': 'ÒõÐÔÔ€²âÖµ(95%CI)',
                                       'F1·ÖÊý': 'F1·ÖÊý(95%CI)', 'Kappa': 'Kappa(95%CI)'}, inplace=True)
    df_dict = {
        '¶àÄ£ÐÍ·ÖÀà-ÑµÁ·Œ¯œá¹û»ã×Ü': df_train_result.drop(['AUC'], axis=1),
        '¶àÄ£ÐÍ·ÖÀà-ÑéÖ€Œ¯œá¹û»ã×Ü': df_test_result.drop(['AUC'], axis=1),
    }
    if delong:
        df_dict.update({'delongŒì²âZÖµŸùÖµ±í': delong_zz})
        df_dict.update({'delongŒì²âPÖµŸùÖµ±í': delong_pp})

    if boostrap != 1:
        plot_name_dict = {
            'ÑµÁ·Œ¯ROCÇúÏßÍŒ': plot_name_list[0],
            'ÑéÖ€Œ¯ROCÇúÏßÍŒ': plot_name_list[1],
            'ÑéÖ€Œ¯¶àÄ£ÐÍÉ­ÁÖÍŒ': plot_name_list[2],
            'ÑéÖ€Œ¯¶àÄ£ÐÍÐ£×ŒÇúÏß': plot_name_list[3],
            'ÑéÖ€Œ¯DCAÇúÏßÍŒ': plot_name_list[4],
        }
    else:
        plot_name_dict = {
            'ÑµÁ·Œ¯ROCÇúÏßÍŒ': plot_name_list[0],
            'ÑéÖ€Œ¯ROCÇúÏßÍŒ': plot_name_list[1],
            'ÑéÖ€Œ¯¶àÄ£ÐÍÐ£×ŒÇúÏß': plot_name_list[2],
            'ÑéÖ€Œ¯DCAÇúÏßÍŒ': plot_name_list[3],
        }

    result_dict = {'str_result': {'·ÖÎöœá¹ûÃèÊö': str_result}, 'tables': df_dict,
                   'pics': plot_name_dict, 'save_pics': plot_name_dict_save}
    return result_dict



def featrueSelect(df, group, features, method='LassoCV', selectNum=None, standardization=False, savePath=None, dpi=600,
                  picFormat='jpeg', decimal_num=3):
    """
    :param df: dataframe  ÕûÌåÊýŸÝ
    :param group:  str  Òò±äÁ¿
    :param features: list Òò×Ó
    :param method:  str  ·œ·š
    :param standardization: ÊÇ·ñ±ê×Œ»¯
    :param savePath:
    :param dpi:
    :param picFormat:
    :param decimal_num:
    :return:
    """

    plot_name_dict, plot_name_dict_save = {}, {}
    str_result = ""
    result_fea, dropFeature = [], []
    df_tab = None
    str_time = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    df_temp = df[features + [group]].dropna().reset_index().drop(columns='index')

    # if df_temp.shape[0] * df_temp.shape[1] > 200000:
    #     return {'error': 'ÊýŸÝÁ¿¹ýŽó£ºÑù±ŸÊýÎª' + str(df_temp.shape[0]) + '£¬Òò×ÓÊý£º' + str(df_temp.shape[1]) + '¡£ÇëŒõÉÙÑù±ŸÁ¿»òÕßŒõÉÙÒò×ÓÊýÁ¿£¡'}

    if standardization:
        rresult_dict = data_standardization(df_temp, features, method='StandardScaler')
        _, df_temp, _, _, _, _ = _analysis_dict(rresult_dict)

    if method == 'LassoCV':
        result_method = LassoCV(cv=5, random_state=0).fit(df_temp[features], df_temp[group])
        coef = pd.Series(result_method.coef_, index=features)
        dropFeature = list(coef[coef == 0].index)
        result_fea = list(coef[coef != 0].index)
        # if not searchFeature:
        #     if len(coef)-len(dropFeature) > searchNum:
        #         imp_coef_fea = list(abs(coef[coef != 0]).sort_values().head(len(coef)-len(dropFeature)-searchNum).index)
        #         dropFeature.extend(imp_coef_fea)
        imp_coef = coef.drop(dropFeature)
        # str_result = 'Íš¹ý'+method+"·œ·šÌÞ³ýµÄÒò×ÓÓÐ£º"+str(dropFeature)+"¡£"
        str_result = 'Íš¹ý' + method + "·œ·š¹²Ñ¡Ôñ³ö" + str(sum(coef != 0)) + "žöµÄÒò×Ó·Ö±ðÎª£º" + str(
            result_fea) + ",Æä×îÓÅ»¯ÕýÔò²ÎÊýÎª£º" + str(round_dec(float(result_method.alpha_), decimal_num)) + "¡£"
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        imp_coef.plot(kind="barh")
        plt.title("Coefficients in the Lasso Model")
        plot_name_dict['Coefficients'] = save_fig(savePath, 'Lasso', 'png', fig, str_time=str_time)
        plot_name_dict_save['Coefficients'] = save_fig(savePath, 'Lasso', picFormat, fig, str_time=str_time)
        plt.close()
    elif method == 'REFCV':

        model = XGBClassifier()
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
                      scoring='accuracy')
        rfecv.fit(df_temp[features], df_temp[group])
        fea_support = list(rfecv.support_)
        for i in range(len(features)):
            if fea_support[i]:
                result_fea.append(features[i])
            else:
                dropFeature.append(features[i])
        if rfecv.n_features_ != len(result_fea):
            return {'error': 'ÊýŸÝŽŠÀíŽíÎó£¬ÇëÖØÐÂÔËÐÐ£¡---REF'}
        str_result = 'Íš¹ý' + method + "·œ·š¹²Ñ¡Ôñ³ö" + str(rfecv.n_features_) + "žöµÄÒò×Ó·Ö±ðÎª£º" + str(result_fea) + "¡£"
        # Plot number of features VS. cross-validation scores
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plot_name_dict['REFÌØÕ÷ÊýÁ¿'] = save_fig(savePath, 'REF', 'png', fig, str_time=str_time)
        plot_name_dict_save['REFÌØÕ÷ÊýÁ¿'] = save_fig(savePath, 'REF', picFormat, fig, str_time=str_time)
        plt.close()
    elif method == 'SVMREFCV':

        model = SVC(kernel="linear")
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
                      scoring='accuracy')
        rfecv.fit(df_temp[features], df_temp[group])
        fea_support = list(rfecv.support_)
        for i in range(len(features)):
            if fea_support[i]:
                result_fea.append(features[i])
            else:
                dropFeature.append(features[i])
        if rfecv.n_features_ != len(result_fea):
            return {'error': 'ÊýŸÝŽŠÀíŽíÎó£¬ÇëÖØÐÂÔËÐÐ£¡---REF'}
        str_result = 'Íš¹ý' + method + "·œ·š¹²Ñ¡Ôñ³ö" + str(rfecv.n_features_) + "žöµÄÒò×Ó·Ö±ðÎª£º" + str(result_fea) + "¡£"
        # Plot number of features VS. cross-validation scores
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plot_name_dict['REFÌØÕ÷ÊýÁ¿'] = save_fig(savePath, 'REF', 'png', fig, str_time=str_time)
        plot_name_dict_save['REFÌØÕ÷ÊýÁ¿'] = save_fig(savePath, 'REF', picFormat, fig, str_time=str_time)
        plt.close()
    elif method == 'PCA':
        if (not selectNum) or selectNum == "":
            pca = PCA(n_components=0.9)
        else:
            if int(selectNum) <= len(features) and int(selectNum) >= 1:
                pca = PCA(n_components=int(selectNum))
            else:
                return {'error': 'Òò×ÓÊýÄ¿ÓŠÔÚ1µœ' + str(len(features)) + 'Ö®Œä£¬ÇëÖØÐÂÊäÈëÉžÑ¡µÄÒò×ÓÊýÄ¿£¡'}
        pca.fit(df_temp[features])
        result_value = np.dot(np.array(df_temp[features]), np.array(pd.DataFrame(pca.components_).T))
        pca1 = PCA(n_components=len(features))
        pca1.fit(df_temp[features])
        dropFeature = features
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        pca_feas = []
        for i in range(pca.n_components_):
            pca_feas.append('PCA' + str(i + 1))
        df_result = pd.DataFrame(result_value, columns=pca_feas)
        df_tab = pd.concat([df_tab, df_result], axis=1)
        if (not selectNum) or selectNum == "":
            str_result = 'Íš¹ý' + method + "·œ·šµ±Ö÷³É·ÖÕŒ±È90%ÒÔÉÏÊ±£¬Òò×ÓœµÎ¬Îª" + str(pca.n_components_) + "Î¬£¬ÆäÐÂµÄÒò×ÓÎª£º" + str(
                pca_feas) + '¡£'
        else:
            str_result = 'Íš¹ý' + method + "·œ·šµ±œµÎ¬Îª" + str(selectNum) + "Ê±" + "£¬ÆäÐÂµÄÒò×ÓÎª£º" + str(pca_feas) + '¡£'
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        plt.title('Scree Plot')
        plt.xlabel("'Factors")
        plt.ylabel("Eigenvalue")
        plt.plot(range(1, len(features) + 1), pca1.explained_variance_)
        plot_name_dict['PCA'] = save_fig(savePath, 'PCA', 'png', fig, str_time=str_time)
        plot_name_dict_save['PCA'] = save_fig(savePath, 'PCA', picFormat, fig, str_time=str_time)
        plt.close()
    elif method == 'mRMR':
        score_features, _, _, score = mrmr_classif(df_temp, X=features, y=group, K=len(features), return_scores=True)
        if (not selectNum) or selectNum == "":
            result_fea = mrmr_classif(df_temp, X=features, y=group, K=score.index(max(score)) + 1, return_scores=False)
        else:
            if int(selectNum) <= len(features) and int(selectNum) >= 1:
                result_fea = mrmr_classif(df_temp, X=features, y=group, K=int(selectNum), return_scores=False)
            else:
                return {'error': 'Òò×ÓÊýÄ¿ÓŠÔÚ1µœ' + str(len(features)) + 'Ö®Œä£¬ÇëÖØÐÂÊäÈëÉžÑ¡µÄÒò×ÓÊýÄ¿£¡'}
        dropFeature = score_features
        for fea in result_fea:
            dropFeature.remove(fea)
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        str_result = 'Íš¹ý' + method + "·œ·š¹²Ñ¡Ôñ³ö" + str(len(result_fea)) + "žöµÄÒò×Ó·Ö±ðÎª£º" + str(result_fea) + "¡£"
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        plt.xlabel("Number of features selected")
        plt.ylabel("mRMR Score")
        plt.plot(range(1, len(features) + 1), score)
        plot_name_dict['mRMR'] = save_fig(savePath, 'mRMR', 'png', fig, str_time=str_time)
        plot_name_dict_save['mRMR'] = save_fig(savePath, 'mRMR', picFormat, fig, str_time=str_time)
        plt.close()
    elif method == 'ReliefF':
        score = []
        ff_features = []

        if (not selectNum) or selectNum == "":
            fs = ReliefF(n_features_to_keep=len(features))
            fs.fit(np.array(df_temp[features]), np.array(df_temp[group]))
            for i in fs.top_features:
                score.append(fs.feature_scores[i])
                ff_features.append(features[i])
            # result_fea = ff_features[0:score.index(max(score))+1]
            for fs in range(len(score)):
                if score[fs] >= 0:
                    result_fea.append(ff_features[fs])
        else:
            if int(selectNum) <= len(features) and int(selectNum) >= 1:
                fs = ReliefF(n_features_to_keep=int(selectNum))
                fs.fit(np.array(df_temp[features]), np.array(df_temp[group]))
                for i in fs.top_features:
                    score.append(fs.feature_scores[i])
                    ff_features.append(features[i])
                result_fea = ff_features[0:int(selectNum)]
            else:
                return {'error': 'Òò×ÓÊýÄ¿ÓŠÔÚ1µœ' + str(len(features)) + 'Ö®Œä£¬ÇëÖØÐÂÊäÈëÉžÑ¡µÄÒò×ÓÊýÄ¿£¡'}
        dropFeature = ff_features
        for fea in result_fea:
            dropFeature.remove(fea)
        if dropFeature != []:
            df_tab = df.drop(dropFeature, axis=1)
        else:
            df_tab = df
        str_result = 'Íš¹ý' + method + "·œ·š¹²Ñ¡Ôñ³ö" + str(len(result_fea)) + "žöµÄÒò×Ó·Ö±ðÎª£º" + str(result_fea) + "¡£"
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        plt.xlabel("Number of features selected")
        plt.ylabel("ReliefF Score")
        plt.plot(range(1, len(features) + 1), score)
        plot_name_dict['ReliefF'] = save_fig(savePath, 'ReliefF', 'png', fig, str_time=str_time)
        plot_name_dict_save['ReliefF'] = save_fig(savePath, 'ReliefF', picFormat, fig, str_time=str_time)
        plt.close()

    result_dict = {'str_result': {'·ÖÎöœá¹ûÃèÊö': str_result}, 'tables': df_tab,
                   'pics': plot_name_dict, 'save_pics': plot_name_dict_save}
    return result_dict
