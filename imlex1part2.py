from matplotlib.pyplot import plot, legend, show, xlabel, ylabel, title
from numpy import dot, transpose
import numpy
from numpy.linalg import pinv
from pandas import read_csv, get_dummies

file_name = 'kc_house_data.csv'


def pre_processing(filename):
    """
    :param filename: path to the data to analyze
    :return: processed data
    """
    data = read_csv(file_name)
    data.dropna(inplace=True)
    remove_index = []
    for i, row in data.iterrows():
        if row['price'] < 1 or row['sqft_lot15'] < 1:
            remove_index.append(i)
    data.drop(["id", "date"], axis=1, inplace=True)
    data.drop(remove_index, inplace=True)
    data = get_dummies(data, columns=['zipcode'])
    noise = [1 for i in range(len(data))]
    data.insert(loc=0, column="noise", value=noise)
    return data


def error_calculation(original, predicted):
    return numpy.array((original - predicted)**2).mean()


def house_pricing_model():
    processed_data = pre_processing(file_name)
    learn_error = []
    test_error = []
    for i in range(1, 100):
        percent_of_learning_sample = i/100
        learn_data = processed_data.sample(frac=percent_of_learning_sample)
        test_data = processed_data.drop(learn_data.index)

        learn_prices = learn_data['price']
        test_prices = test_data['price']

        learn_data.drop(['price'], 1, inplace=True)
        test_data.drop(['price'], 1, inplace=True)

        x_dagger = pinv(transpose(learn_data.as_matrix()))
        w_hat = dot(transpose(x_dagger), learn_prices)

        learn_data = learn_data.as_matrix()
        test_data = test_data.as_matrix()

        learn_result = dot(learn_data, w_hat)

        learn_error.append(error_calculation(learn_prices, learn_result))
        test_result = dot(test_data, w_hat)
        test_error.append(error_calculation(test_prices, test_result))
    # graph_plotting(learn_error, test_error)
    index = [i for i in range(1, 100)]
    plot(index, learn_error, label="learn error")
    plot(index, test_error, label="test error")
    xlabel("percentage devoted to training")
    ylabel("error value")
    title("House pricing model - error values")
    legend()
    show()

if __name__ == '__main__':
    house_pricing_model()
