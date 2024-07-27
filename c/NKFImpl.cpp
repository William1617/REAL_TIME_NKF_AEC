
#include "NKFImpl.h"

void NKFImpl::ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}

void NKFImpl::Enhance(std::string in_audio,std::string lpb_audio,std::string out_audio){
	std::vector<float>  testdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputfile;
	inputfile.load(in_audio);
	AudioFile<float> inputlpbfile;
	inputlpbfile.load(lpb_audio);

	int audiolen=inputfile.getNumSamplesPerChannel();
	int audiolen2=inputlpbfile.getNumSamplesPerChannel();
	audiolen =audiolen2<audiolen ? audiolen2:audiolen;
    int process_num=audiolen/BLOCK_SHIFT;

	for(int i=0;i<process_num;i++){
        memmove(m_pEngine.mic_buffer, m_pEngine->mic_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
		memmove(m_pEngine.lpb_buffer, m_pEngine->lpb_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
      
        for(int n=0;n<BLOCK_SHIFT;n++){
            m_pEngine->mic_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputfile.samples[0][n+i*BLOCK_SHIFT];
			m_pEngine->lpb_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputlpbfile.samples[0][n+i*BLOCK_SHIFT];
		} 
        OnnxInfer();
        for(int j=0;j<BLOCK_SHIFT;j++){
            testdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV(out_audio,testdata,SAMEPLERATE);
}


void NKFImpl::OnnxInfer() {

	float estimated_block[BLOCK_LEN]={0};
    float mic_real[FFT_OUT_SIZE]={0};
    float mic_imag[FFT_OUT_SIZE]={0};
    
    double mic_in[BLOCK_LEN]={0};
    std::vector<cpx_type> mic_res(BLOCK_LEN);
    double lpb_in[BLOCK_LEN]={0};
    std::vector<cpx_type> lpb_res(BLOCK_LEN);

	std::vector<size_t> shape;
    shape.push_back(BLOCK_LEN);
    std::vector<size_t> axes;
    axes.push_back(0);
    std::vector<ptrdiff_t> stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));

	for (int i = 0; i < BLOCK_LEN; i++){
        mic_in[i] = m_pEngine.mic_buffer[i]*m_windows[i];
        lpb_in[i]= m_pEngine.lpb_buffer[i]*m_windows[i];
	}
   
    pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, mic_in, mic_res.data(), 1.0);
    pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, lpb_in,lpb_res.data(), 1.0);

    memmove(m_pEngine.lpb_real,m_pEngine.lpb_real+FFT_OUT_SIZE,(NKF_LEN-1)*FFT_OUT_SIZE*sizeof(float));
    memmove(m_pEngine.lpb_imag,m_pEngine.lpb_imag+FFT_OUT_SIZE,(NKF_LEN-1)*FFT_OUT_SIZE*sizeof(float));
    for (int i=0;i<FFT_OUT_SIZE;i++){
        m_pEngine.lpb_real[(NKF_LEN-1)*FFT_OUT_SIZE+i]=static_cast<float>(lpb_res[i].real());
        m_pEngine.lpb_imag[(NKF_LEN-1)*FFT_OUT_SIZE+i]=static_cast<float>(lpb_res[i].imag());
        mic_real[i]=static_cast<float>(mic_res[i].real());
        mic_imag[i]=static_cast<float>(mic_res[i].imag());
    }
    float dh_real[NKF_LEN*FFT_OUT_SIZE]={0};
    float dh_imag[NKF_LEN*FFT_OUT_SIZE]={0};
    for (int i=0;i<NKF_LEN*FFT_OUT_SIZE;i++){
        dh_real[i]=static_cast<float>(m_pEngine.h_posterior_real[i]-m_pEngine.h_prior_real[i]);
        dh_imag[i]=static_cast<float>(m_pEngine.h_posterior_imag[i]-m_pEngine.h_prior_imag[i]);

    }

	memcpy(m_pEngine.h_prior_real,m_pEngine.h_posterior_real,NKF_LEN*FFT_OUT_SIZE*sizeof(double));
    memcpy(m_pEngine.h_prior_imag,m_pEngine.h_posterior_imag,NKF_LEN*FFT_OUT_SIZE*sizeof(double));

    float input_feature_real[(2*NKF_LEN+1)*FFT_OUT_SIZE]={0};
    float input_feature_imag[(2*NKF_LEN+1)*FFT_OUT_SIZE]={0};
    double e_real[FFT_OUT_SIZE]={0};
    double e_imag[FFT_OUT_SIZE]={0};
    

    int k=2*NKF_LEN+1;
    int is_tensor=1;
    for (int i=0;i<FFT_OUT_SIZE;i++){
       
        for (int j=0;j<NKF_LEN;j++){
            input_feature_real[k*i+j]=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i];
            input_feature_imag[k*i+j]=m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i];
            input_feature_real[k*i+j+NKF_LEN+1]=dh_real[NKF_LEN*i+j];
            input_feature_imag[k*i+j+NKF_LEN+1]=dh_imag[NKF_LEN*i+j];

            e_real[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_real[NKF_LEN*i+j] -m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_imag[NKF_LEN*i+j];
            e_imag[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_imag[NKF_LEN*i+j] +m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_real[NKF_LEN*i+j];
        }
        e_real[i]=mic_real[i]-e_real[i];
        e_imag[i]=mic_imag[i]-e_imag[i];
        input_feature_real[k*i+NKF_LEN]=static_cast<float>(e_real[i]);
        input_feature_imag[k*i+NKF_LEN]=static_cast<float>(e_imag[i]);
        
    }

    
	
	ort_inputs.clear();
	ort_inputs.resize(6);

	ort_inputs[0] = Ort::Value::CreateTensor<float>(
		memory_info, input_feature_real, FFT_OUT_SIZE*k, infea_node_dims, 3);
	
	ort_inputs[1] = Ort::Value::CreateTensor<float>(
		memory_info, input_feature_imag, FFT_OUT_SIZE*k, infea_node_dims, 3);
	

	for (int i=0;i<4;i++){
		ort_inputs[2+i]=Ort::Value::CreateTensor<float>(
		memory_info,  m_pEngine.instates[i].data(),18*FFT_OUT_SIZE, in_states_dims, 3);
	}
	
    ort_outputs = session->Run(Ort::RunOptions{nullptr},
		input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
		output_node_names.data(), output_node_names.size());
    
	float *kg_real = ort_outputs[0].GetTensorMutableData<float>();
	float *kgimag = ort_outputs[1].GetTensorMutableData<float>();

	float *out_state;
	for(int i=0;i<4;i++){
		out_state=ort_outputs[i+2].GetTensorMutableData<float>();
		memcpy(m_pEngine.instates[i].data(), out_state, FFT_OUT_SIZE*18 * sizeof(float));
	}
	for (int i=0;i<FFT_OUT_SIZE;i++){
        for (int j=0;j<NKF_LEN;j++){
            m_pEngine.h_posterior_real[NKF_LEN*i+j] =m_pEngine.h_prior_real[NKF_LEN*i+j] +e_real[i]*kgreal[NKF_LEN*i+j]-e_imag[i]*kgimag[NKF_LEN*i+j];
            m_pEngine.h_posterior_imag[NKF_LEN*i+j] =m_pEngine.h_prior_imag[NKF_LEN*i+j] +e_imag[i]*kgreal[NKF_LEN*i+j]+e_real[i]*kgimag[NKF_LEN*i+j];

        }
    }

    double echohat_real[FFT_OUT_SIZE]={0};
    double echohat_imag[FFT_OUT_SIZE]={0};

    for (int i=0;i<FFT_OUT_SIZE;i++){
        for (int j=0;j<NKF_LEN;j++){
            echohat_real[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_posterior_real[NKF_LEN*i+j] -m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_posterior_imag[NKF_LEN*i+j];
            echohat_imag[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_posterior_imag[NKF_LEN*i+j] +m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_posterior_real[NKF_LEN*i+j];

        }
        mic_res[i] = cpx_type(mic_real[i]-echohat_real[i],mic_imag[i]-echohat_imag[i]);
    }

    pocketfft::c2r(shape, strideo, stridel, axes, pocketfft::BACKWARD, mic_res.data(), mic_in, 1.0); 

	for (int i = 0; i < FFT_OUT_SIZE; i++) {
        mic_res[i] = cpx_type(output_fea[2*i] , output_fea[2*i+1]);
	}
    pocketfft::c2r(shape, strideo, stridel, axes, pocketfft::BACKWARD, mic_res.data(), mic_in, 1.0);   
    
    for (int i = 0; i < BLOCK_LEN; i++)
        estimated_block[i] = mic_in[i] / BLOCK_LEN;   

	memmove(m_pEngine->out_buffer, m_pEngine->out_buffer + BLOCK_SHIFT, 
        (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine->out_buffer + (BLOCK_LEN - BLOCK_SHIFT), 
        0, BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < BLOCK_LEN; i++){
        m_pEngine->out_buffer[i] += estimated_block[i]*m_windows[i];
    }
   
}
 
