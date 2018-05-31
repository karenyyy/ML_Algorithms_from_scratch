//
// Created by karen on 5/30/18.
//

#ifndef ML_ALGORITHMS_FROM_SCRATCH_LOGISTIC_REGRESSION_H
#define ML_ALGORITHMS_FROM_SCRATCH_LOGISTIC_REGRESSION_H


#include <string>
#include <memory>
#include <vector>

using namespace std;

namespace ANN {

    template<typename T>
    class LogisticRegression {
    public:
        LogisticRegression() = default;
        int init(const T* data,
                 const T* labels,
                 int train_num,
                 int feature_length,
                 int reg_kinds = -1,
                 T learning_rate = 0.00001, int iterations = 10000, int train_method = 0, int mini_batch_size = 1);
        int train(const string& model);
        int load_model(const string& model);

        // const after a function declaration means that the function is not allowed to change any class members
        // like read-only functions
        T predict(const T* data, int feature_length) const; 

        // Regularization kinds
        enum RegKinds {
            REG_DISABLE = -1, // Regularization disabled
            REG_L1 = 0 // L1 norm
        };

        // Training methods
        enum Methods {
            BATCH = 0,
            MINI_BATCH = 1
        };

    private:
        int store_model(const string& model) const;
        T calc_sigmoid(T x) const; // y = 1/(1+exp(-x))
        T norm(const vector<T>& v1, const vector<T>& v2) const;
        void batch_gradient_descent();
        void mini_batch_gradient_descent();
        void gradient_descent(const vector<vector<T>>& data_batch, const vector<T>& labels_batch, int length_batch);

        vector<vector<T>> data;
        vector<T> labels;
        int iterations = 1000;
        int train_num = 0; // train samples num
        int feature_length = 0;
        T learning_rate = 0.00001;
        vector<T> thetas; // coefficient
        T epsilon = 0.000001; // termination condition
        T lambda = (T)0.; // regularization method
        int train_method = 0;
        int mini_batch_size = 1;
    };

} // namespace ANN




#endif //ML_ALGORITHMS_FROM_SCRATCH_LOGISTIC_REGRESSION_H
