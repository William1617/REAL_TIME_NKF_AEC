
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#include "pocketfft_hdronly.h"
#include "AudioFile.h"



#define SAMEPLERATE  (16000)
#define BLOCK_LEN		(512)
#define BLOCK_SHIFT  (256)
#define FFT_OUT_SIZE (257)
#define NKF_LEN (4)
typedef complex<double> cpx_type;

struct nkf_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };
    float out_buffer[BLOCK_LEN] = { 0 };
    float lpb_buffer[BLOCK_LEN]= {0};

    float lpb_real[FFT_OUT_SIZE*NKF_LEN]={0};
    float lpb_imag[FFT_OUT_SIZE*NKF_LEN]={0};
    double h_prior_real[FFT_OUT_SIZE*NKF_LEN]={0};
    double h_prior_imag[FFT_OUT_SIZE*NKF_LEN]={0};
    double h_posterior_real[FFT_OUT_SIZE*NKF_LEN]={0};
    double h_posterior_imag[FFT_OUT_SIZE*NKF_LEN]={0};

    std::vector<std::vector<float>> instates;
    
};

class NKFImpl{
public:

    int Enhance(std::string in_audio,std::string lpb_audio,std::string out_audio);
    

private:
    void init_engine_threads(int inter_threads, int intra_threads){
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void init_onnx_model(const std::string ModelPath){
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, ModelPath.c_str(), session_options);
    };
    void ResetInout(){

        m_pEngine.instates.clear();
        m_pEngine.instates.resize(4);
        for (int i=0;i<4;i++){
            m_pEngine.instates[i].clear();
            m_pEngine.instates[i].resize(FFT_OUT_SIZE*18);
            std::fill(m_pEngine.instates[i].begin(),m_pEngine.instates[i].end(),0);
        }
        memset(m_pEngine.mic_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.lpb_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.out_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.lpb_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(float));
        memset(m_pEngine.lpb_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(float));
        memset(m_pEngine.h_posterior_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_posterior_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_prior_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_prior_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));

    };
    void ExportWAV(const std::string & Filename, 
		const std::vector<float>& Data, unsigned SampleRate);
    void OnnxInfer();

    
public:
    NKFImpl(const std::string ModelPath){
    init_onnx_model(ModelPath);
    for (int i=0;i<BLOCK_LEN;i++){
        m_windows[i]=sinf(PI*i/(BLOCK_LEN-1));
    }
    ResetInout();
   }

private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    nkf_engine m_pEngine;
    std::vector<Ort::Value> ort_inputs;
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> input_node_names = {"in_real","in_imag","in_hrr","in_hir","in_hri","in_hii"};

	std::vector<const char *> output_node_names = {"enh_real","enh_imag","out_hrr","out_hir","out_hri","out_hii"};

    const int64_t infea_node_dims[3] = {FFT_OUT_SIZE,1,2*NKF_LEN+1}; 
	const int64_t in_states_dims[3] = {1,FFT_OUT_SIZE,18};

    float m_windows[BLOCK_LEN]={0};

};
