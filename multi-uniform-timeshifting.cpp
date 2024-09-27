/*
Multi-uniform-timeshifting
contact: Dr. Jonnel A. Jaurigue
https://orcid.org/0009-0003-0802-076X
*/


#include "armadillo-12.2.0/include/armadillo"
#include <iomanip> // Library for setting fixed decimal places in cout.
using namespace std;

int main(){
    // ----------  Parameters initialisation
    // !!! user sets these values !!!
    int buffer_pretraining = 500;
    int buffer_pretesting = 500;
    int k_training_inputsteps = 6000;
    int k_testing_inputsteps = 3000;
    int replicas_N = 5;
    int S_features = 325; // number of features of the base state matrix S
    int recall_J = S_features; // Multi-uniform-timeshifting uses total recall, so set recall_J = S_features
    int d2 = 3;
    int d3 = 1;
    int d4 = 1;
    int d5 = 2;
    double Rtik = 0.; // tikhonov/ridge regularisation parameter


    // ---------- Check that there are enough buffer inputs for multi-uniform-timeshifting   
    if(d2 + d3 + d4 + d5 > buffer_pretraining){
        cout << "Sum of StepsizeLayers > buffer_pretraining... exiting program...\n";
        exit(EXIT_SUCCESS);
    }


    // ---------- Create the data file to store NRMSE values
    stringstream NRMSEfilename;
    NRMSEfilename.precision(7); 
    NRMSEfilename 
    << "NRMSE"
    << "_" << "O" << "_" << S_features + (replicas_N - 1)*(S_features)
    << "_" << "N" << "_" << replicas_N
    << "_" << "J" << "_" << recall_J << "_of_" << S_features
    << "_" << "d2" << "_" << d2   
    << "_" << "d3" << "_" << d3
    << "_" << "d4" << "_" << d4
    << "_" << "d5" << "_" << d5
    << ".txt";
    ofstream NRMSEfiledata;
    NRMSEfiledata.open(NRMSEfilename.str().c_str(),ios::trunc);


    // ---------- Import your target series vector y
    int K_inputsteps = buffer_pretraining + k_training_inputsteps + buffer_pretesting + k_testing_inputsteps;
    arma::vec targetseries_y(K_inputsteps, arma::fill::zeros);
    targetseries_y.load("targetseries_y.txt", arma::csv_ascii);


    // ---------- Initialise the final multi-uniform-timeshifts state matrix [S, ..., S_d5] of O features, with bias column
    arma::mat finalstatematrix(K_inputsteps + d2 + d3 + d4 + d5, S_features*replicas_N + 1, arma::fill::zeros);


    // ---------- Get the reservoir current state
    // In this case, we will be sourcing reservoir states from an imported hardware-implemented base state matrix S
    double current_state;
    arma::mat base_S(K_inputsteps, S_features, arma::fill::zeros);
    base_S.load("base_S.txt", arma::csv_ascii);


    // ---------- Algorithm for multi-uniform-timeshifting
    // Determine current input-step
    for(unsigned long int i = 0; i < K_inputsteps; ++i){
        // Insert bias
        finalstatematrix(i, S_features) = 1.;
        
        // Determine current mask-step
        for(unsigned long int j = 0; j < S_features; ++j){

            // Get reservoir current state, sourced from our imported base state matrix S
            current_state = base_S(i, j);    
            
            for(int N = 1; N <= replicas_N; ++N
            ){  
                // Insert current state at base state matrix S position of final state matrix
                if(
                    N == 1 && 
                    j < S_features
                ){
                    finalstatematrix(
                    i, j
                    ) = current_state;
                }
                // Insert current state at uniformly-timeshifted state matrix Sd2 position of final state matrix
                if(   
                    N == 2 &&
                    (N - 1)*(S_features - recall_J) <= j &&
                    j <= (N - 1)*(S_features - recall_J) + S_features - 1
                ){
                    finalstatematrix(
                        i + d2,
                        S_features + 1 + (N - 2)*S_features + (j - (N - 1)*(S_features - recall_J))
                    ) = current_state;
                }
                // Insert current state at uniformly-timeshifted state matrix Sd3 position of final state matrix
                if(
                    N == 3 &&
                    (N - 1)*(S_features - recall_J) <= j &&
                    j <= (N - 1)*(S_features - recall_J) + S_features - 1
                ){
                    finalstatematrix(
                        i + d2 + d3,
                        S_features + 1 + (N - 2)*S_features + (j - (N - 1)*(S_features - recall_J))
                    ) = current_state;
                }
                // Insert current state at uniformly-timeshifted state matrix Sd4 position of final state matrix
                if(
                    N == 4 &&
                    (N - 1)*(S_features - recall_J) <= j &&
                    j <= (N - 1)*(S_features - recall_J) + S_features - 1
                ){
                    finalstatematrix(
                        i + d2 + d3 + d4,
                        S_features + 1 + (N - 2)*S_features + (j - (N - 1)*(S_features - recall_J))
                    ) = current_state;
                }
                // Insert current state at uniformly-timeshifted state matrix Sd5 position of final state matrix
                if(
                    N == 5 &&
                    (N - 1)*(S_features - recall_J) <= j &&
                    j <= (N - 1)*(S_features - recall_J) + S_features - 1
                ){
                    finalstatematrix(
                        i + d2 + d3 + d4 + d5,
                        S_features + 1 + (N - 2)*S_features + (j - (N - 1)*(S_features - recall_J))
                    ) = current_state;
                }
            }
        }
    }


    // ---------- Algorithm for training and testing
    // Create training and testing state matrices
    arma::mat trainingstatematrix = finalstatematrix.rows(buffer_pretraining, buffer_pretraining + k_training_inputsteps - 1); 
    arma::mat testingstatematrix = finalstatematrix.rows(buffer_pretraining + k_training_inputsteps + buffer_pretesting, K_inputsteps - 1);

    // Create training and testing target vectors from targetseries_y
    arma::vec targetseries_y_training = targetseries_y.subvec(buffer_pretraining, buffer_pretraining + k_training_inputsteps - 1); 
    arma::vec targetseries_y_testing = targetseries_y.subvec(buffer_pretraining + k_training_inputsteps + buffer_pretesting, K_inputsteps - 1);


    // Train for weights_w using the pseudoinverse algorithm
    arma::mat finalstatematrix_pseudoinverse(S_features*replicas_N + 1, k_training_inputsteps, arma::fill::zeros);
    arma::mat identity_matrix; 
    identity_matrix.eye(S_features*replicas_N + 1, S_features*replicas_N + 1);
    finalstatematrix_pseudoinverse = arma::inv(trainingstatematrix.t()*trainingstatematrix + Rtik*identity_matrix)*trainingstatematrix.t();
    arma::vec weights_w = finalstatematrix_pseudoinverse*targetseries_y_training;

    // Predict targets using the weights_w vector
    arma::vec predicted_targetseries_y_training = trainingstatematrix*weights_w;
    arma::vec predicted_targetseries_y_testing = testingstatematrix*weights_w;

    // Measure performance using NRMSE, normalised by sqrt(variance)
    double NRMSE_training;
    double NRMSE_testing;

    double targetseries_y_training_variance = arma::var(targetseries_y_training);
    for(int i = 0; i < k_training_inputsteps; ++i){
        NRMSE_training = NRMSE_training + ((targetseries_y_training(i) - predicted_targetseries_y_training(i))*(targetseries_y_training(i) - predicted_targetseries_y_training(i)));
    }
    NRMSE_training = NRMSE_training/(targetseries_y_training_variance*k_training_inputsteps);
    NRMSE_training = sqrt(NRMSE_training);
    
    double targetseries_y_testing_variance = arma::var(targetseries_y_testing);
    for(int i = 0; i < k_testing_inputsteps; ++i){
        NRMSE_testing = NRMSE_testing + ((targetseries_y_testing(i) - predicted_targetseries_y_testing(i)) *(targetseries_y_testing(i) - predicted_targetseries_y_testing(i)));
    }
    NRMSE_testing = NRMSE_testing/(targetseries_y_testing_variance*k_testing_inputsteps);
    NRMSE_testing = sqrt(NRMSE_testing);

    NRMSEfiledata << std::setprecision(10) 
    <<'\t'<< NRMSE_training 
    <<'\t'<< NRMSE_testing
    <<'\n';
    NRMSEfiledata.close();
}


/*
COMPILE:
g++ -std=c++11 -llapack -lblas multi-uniform-timeshifting.cpp -o multi-uniform-timeshifting
// llapack and lblas are needed for armadillo

RUN:
./multi-uniform-timeshifting
*/