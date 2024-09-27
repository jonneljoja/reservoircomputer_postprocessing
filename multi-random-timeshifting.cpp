/*
Multi-random-timeshifting
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
    int S_features = 325; // number of features of the base state matrix S
    int desired_O = 1000; // desired feature dimension O
    int R = 10;
    double Rtik = 1.0; // tikhonov/ridge regularisation parameter


    // ---------- ----------  Derived parameters initialisation ---------- ---------- START
    int recalled_states_per_mask = floor(desired_O/S_features);
    int masks_with_extra_recalled_state = desired_O % S_features;


    // ---------- Check that there are enough base state matrix features for the desired feature dimension O
    if(desired_O > (R + 1)*S_features){
        cout << "desired_O > (R + 1)*S_features... exiting program...\n";
        exit(EXIT_SUCCESS);
    }


    // ---------- Check that there are enough buffer inputs for multi-random-timeshifting   
    if(R > buffer_pretraining){
        cout << "R > buffer_pretraining... exiting program...\n"; // SANITY CHECK // Concantenation approach
        exit(EXIT_SUCCESS);
    }


    // ---------- Create the data file to store NRMSE values
    stringstream NRMSEfilename;
    NRMSEfilename.precision(7); 
    NRMSEfilename 
    << "NRMSE"
    << "_" << "O" << "_" << desired_O
    << "_" << "ReservoirDim" << "_" << S_features
    << "_" << "R" << "_" << R
    << ".txt";
    ofstream NRMSEfiledata;
    NRMSEfiledata.open(NRMSEfilename.str().c_str(),ios::trunc);


    // ---------- Import your target series vector y
    int K_inputsteps = buffer_pretraining + k_training_inputsteps + buffer_pretesting + k_testing_inputsteps;
    arma::vec targetseries_y(K_inputsteps, arma::fill::zeros);
    targetseries_y.load("targetseries_y.txt", arma::csv_ascii);


    // ---------- Initialise the final multi-random-timeshifts state matrix [Sr1, ..., S_rN] of O features, with bias column
    arma::mat finalstatematrix(K_inputsteps + R, desired_O + 1, arma::fill::zeros);


    // ---------- Get the reservoir current state
    // In this case, we will be sourcing reservoir states from an imported hardware-implemented base state matrix S
    double current_state;
    arma::mat base_S(K_inputsteps, S_features, arma::fill::zeros);
    base_S.load("base_S.txt", arma::csv_ascii);


    // ---------- Algorithm for multi-random-timeshifting
    // Build random number generator
    std::random_device rd;
    std::mt19937 RNGenerator(rd()); 
    std::uniform_real_distribution<double> uniform_real_distribution(0, 1);

    // Initialise a matrix of random timeshifts r
    arma::mat r_matrix(R + 1, S_features, arma::fill::zeros);
    // Generate the vector of integers from 0 to R to be used for each vector of random-timeshifts r
    arma::vec uniquetimeshifts(R + 1, arma::fill::zeros);
    for(int i_uniquetimeshifts = 0; i_uniquetimeshifts <= R; ++i_uniquetimeshifts){
        uniquetimeshifts(i_uniquetimeshifts) = i_uniquetimeshifts;
    }
    for(int j_r_matrix = 0; j_r_matrix < S_features; ++j_r_matrix){
        // Random column vector to insert
        std::shuffle(uniquetimeshifts.begin(), uniquetimeshifts.end(), RNGenerator);
        for(int i_r_matrix = 0; i_r_matrix <= R; ++i_r_matrix){
            // Leading uniquetimeshifts numbers are the indices that indicate the virtual node sampled for that clock cycle.
            r_matrix(i_r_matrix, j_r_matrix) = uniquetimeshifts(i_r_matrix);
        }
    }

    // Determine current input-step
    for(unsigned long int i = 0; i < K_inputsteps; ++i){
        // Insert bias
        finalstatematrix(i, S_features) = 1.;
        
        // Determine current mask-step
        for(unsigned long int j = 0; j < S_features; ++j){

            // Get reservoir current state, sourced from our imported base state matrix S
            current_state = base_S(i, j);    

            if(
                j < S_features
            ){
                for( 
                    int i_recalled_states_per_mask = 0;
                    i_recalled_states_per_mask < (recalled_states_per_mask);
                    i_recalled_states_per_mask = (i_recalled_states_per_mask + 1)
                ){
                    finalstatematrix(
                            i + r_matrix(i_recalled_states_per_mask, j), 
                            j + i_recalled_states_per_mask*S_features
                        ) = current_state;                   
                }
            }

            if(
                j < masks_with_extra_recalled_state
            ){
                for( 
                    int i_masks_with_extra_recalled_state = 0;
                    i_masks_with_extra_recalled_state < (masks_with_extra_recalled_state);
                    i_masks_with_extra_recalled_state = (i_masks_with_extra_recalled_state + 1)
                ){
                    finalstatematrix(
                            i + r_matrix(recalled_states_per_mask, j), 
                            j + recalled_states_per_mask*S_features
                        ) = current_state;                   
                }
            }
        }
    }


    // ---------- Algorithm for training and testing
    // create training and testing state matrices
    arma::mat trainingstatematrix = finalstatematrix.rows(buffer_pretraining, buffer_pretraining + k_training_inputsteps - 1); 
    arma::mat testingstatematrix = finalstatematrix.rows(buffer_pretraining + k_training_inputsteps + buffer_pretesting, K_inputsteps - 1);

    // create training and testing target output vectors from targetseries_y
    arma::vec targetseries_y_training = targetseries_y.subvec(buffer_pretraining, buffer_pretraining + k_training_inputsteps - 1); 
    arma::vec targetseries_y_testing = targetseries_y.subvec(buffer_pretraining + k_training_inputsteps + buffer_pretesting, K_inputsteps - 1);

    // Train for weights_w using the pseudoinverse algorithm
    arma::mat finalstatematrix_pseudoinverse( desired_O + 1, k_training_inputsteps, arma::fill::zeros);
    arma::mat identity_matrix;
    identity_matrix.eye(desired_O + 1, desired_O + 1);
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
// Main function ---------- END


/*
COMPILE:
g++ -std=c++11 -llapack -lblas multi-random-timeshifting.cpp -o multi-random-timeshifting
// llapack and lblas are needed for armadillo

RUN:
./multi-random-timeshifting
*/