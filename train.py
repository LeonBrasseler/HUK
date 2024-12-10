from glm import train_glm
from glm import evaluate_glm
from NN import train_NN_model
from NN import evaluate_model

if __name__ == '__main__':
    glm_pred, glm_freq, glm_sev, data_test = train_glm()
    metrics_glm = evaluate_glm(glm_freq, glm_sev, data_test)

    classifier, regressor, test_data = train_NN_model()
    metrics_NN = evaluate_model(classifier, regressor, test_data)

    print("MAE for GLM: ", metrics_glm)
    print("MAE for NN: ", metrics_NN)

