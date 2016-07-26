#include "LinearSvmModel.hpp"

#include "detector_model.pb.h"

#include "helpers/Log.hpp"

#include <boost/filesystem.hpp>

/*#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/bind.hpp>
*/

#include <stdexcept>
#include <fstream>
#include <iterator>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "LinearSvmModel");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "LinearSvmModel");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "LinearSvmModel");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;
using namespace boost;

//namespace qi = boost::spirit::qi;

LinearSvmModel::LinearSvmModel()
{
    // w is left empty
    bias = 0;
    return;
}

LinearSvmModel::LinearSvmModel(const doppia_protobuf::DetectorModel &model)
{

    if(model.has_detector_name())
    {
        log_info() << "Parsing model " << model.detector_name() << std::endl;
    }

    if(model.detector_type() != doppia_protobuf::DetectorModel::LinearSvm)
    {
        throw std::runtime_error("Received model is not of the expected type LinearSvm");
    }

    if(model.has_linear_svm_model() == false)
    {
        throw std::runtime_error("The model content does not match the model type");
    }

    throw std::runtime_error("LinearSvmModel from doppia_protobuf::DetectorModel not yet implemented");

    return;
}

LinearSvmModel::LinearSvmModel(const Eigen::VectorXf &w_, const float bias_)
{
    w = w_;
    bias = bias_;
    return;
}

LinearSvmModel::LinearSvmModel(const std::string &filename)
{

    if(filesystem::exists(filename) == false)
    {
        log_error() << "Could not find the linear SVM model file " << filename << std::endl;
        throw std::invalid_argument("Could not find linear SVM model file");
    }

    log_info() << "Parsing linear svm model " << filename << std::endl;

    // open and parse the file
    std::ifstream model_file(filename.c_str());
    if(model_file.is_open() == false)
    {
        log_error() << "Could not open the linear SVM model file " << filename << std::endl;
        throw std::runtime_error("Failed to open the  linear SVM model file");
    }
    parse_libsvm_model_file(model_file);
    return;
}

LinearSvmModel::~LinearSvmModel()
{
    // nothing to do here
    return;
}


float LinearSvmModel::get_bias() const
{
    return bias;
}

const Eigen::VectorXf &LinearSvmModel::get_w() const
{
    return w;
}


void hardcoded_parsing(std::ifstream &model_file, float &bias, Eigen::VectorXf &w)
{
    //solver_type L2R_L2LOSS_SVC_DUAL
    //nr_class 2
    //label 1 -1
    //nr_feature 5120
    //bias -1
    //w
    //-9.928359971228299e-05
    // -1.046861188325508e-05
    // ...


    string t_string, solver_type;
    int nr_class, nr_feature;
    vector<int> labels;

    model_file >> t_string >> solver_type;
    model_file >> t_string >> nr_class;
    labels.resize(nr_class);
    model_file >> t_string; // label
    for(int i=0;i < nr_class; i+=1)
    {
        model_file >> labels[i];
    }
    model_file >> t_string >> nr_feature;
    w.setZero(nr_feature);
    model_file >> t_string >> bias;
    model_file >> t_string; // w
    for(int i=0;i < nr_feature; i+=1)
    {
        model_file >> w(i);
    }

    log_debug() << "Read a model trained using == " << solver_type << std::endl;
    log_debug() << "Read bias == " << bias << std::endl;
    log_debug() << "Read w of size " << w.size() << std::endl; // << w << std::endl;

    return;
}


void LinearSvmModel::parse_libsvm_model_file(std::ifstream &model_file)
{

    //parse_using_boost_spirit(model_file, bias, w);
    hardcoded_parsing(model_file, bias, w);

    return;
}

float LinearSvmModel::compute_score(const Eigen::VectorXf &feature_vector) const
{
    assert(w.size() > 0);
    return feature_vector.dot(w) - bias;
}

} // end of namespace doppia
