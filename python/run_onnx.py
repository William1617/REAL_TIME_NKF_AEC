import onnxruntime
import soundfile as sf
import numpy as np
block_len=1024
block_shift=256
fft_len=block_len//2+1
nkf=onnxruntime.InferenceSession('./nkfsim.onnx')
L=4

model_input_name= [inp.name for inp in nkf.get_inputs()]
model_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in nkf.get_inputs()}

windows=[]
for idx1 in range (block_len):
    windows.append(np.sin(np.pi*idx1/(block_len-1))*np.sin(np.pi*idx1/(block_len-1)))

def run_nkf(mic_audio,ref_audio):
    audio_len=min(len(mic_audio),len(ref_audio))
    frame_num=audio_len//block_shift -4
   
    xt_real=np.zeros((fft_len,1,L))
    xt_imag=np.zeros((fft_len,1,L))
    h_prior_real = np.zeros((fft_len,4,1))
    h_prior_imag = np.zeros((fft_len,4,1))
    h_posterior_real=np.zeros((fft_len,4,1))
    h_posterior_imag=np.zeros((fft_len,4,1))
    out_audio=np.zeros(audio_len)
    mic_frame=np.zeros(block_len)
    ref_frame=np.zeros(block_len)

    y_real=np.zeros((fft_len,1,1))
    y_imag=np.zeros((fft_len,1,1))

    for idx in range(frame_num):
        mic_frame[:-block_shift]=mic_frame[block_shift:]
        ref_frame[:-block_shift]=ref_frame[block_shift:]
        mic_frame[-block_shift:]=mic_audio[idx*block_shift:idx*block_shift+block_shift]
        ref_frame[-block_shift:]=ref_audio[idx*block_shift:idx*block_shift+block_shift]

        ref_fft=np.fft.rfft(ref_frame*windows)
       
        ref_real=np.real(ref_fft)
        ref_imag=np.imag(ref_fft)

        mic_fft=np.fft.rfft(mic_frame*windows)
        mic_real=np.real(mic_fft)
        mic_imag=np.imag(mic_fft)

        dh_real=(h_posterior_real-h_prior_real).squeeze()
        dh_imag=(h_posterior_imag-h_prior_imag).squeeze()
        dh_real=np.expand_dims(dh_real,axis=1)
        dh_imag=np.expand_dims(dh_imag,axis=1)
        
        h_prior_real=h_posterior_real.copy()
        h_prior_imag=h_posterior_imag.copy()

        xt_real[:,:,:-1]=xt_real[:,:,1:]
        xt_real[:,0,-1]=ref_real
        xt_imag[:,:,:-1]=xt_imag[:,:,1:]
        xt_imag[:,0,-1]=ref_imag

        y_real[:,0,0]=mic_real
        y_imag[:,0,0]=mic_imag
    
        e_real=y_real-np.matmul(xt_real,h_prior_real)+np.matmul(xt_imag,h_prior_imag)
        e_imag=y_imag-np.matmul(xt_imag,h_prior_real)-np.matmul(xt_real,h_prior_imag)
        input_real=np.concatenate([xt_real,e_real,dh_real],axis=-1)
        input_imag=np.concatenate([xt_imag,e_imag,dh_imag],axis=-1)

        model_input[model_input_name[0]] = input_real.astype('float32')
        model_input[model_input_name[1]] = input_imag.astype('float32')
    
        model_output = nkf.run(None, model_input)
        kg_real=model_output[0].squeeze()
        kg_imag=model_output[1].squeeze()
        kg_real=np.expand_dims(kg_real,axis=-1)
        kg_imag=np.expand_dims(kg_imag,axis=-1)

        for idx3 in range(2,6):
            model_input[model_input_name[idx3]] = model_output[idx3]
        
        
        h_posterior_real=h_prior_real+np.matmul(kg_real,e_real)-np.matmul(kg_imag,e_imag)
        h_posterior_imag=h_prior_imag+np.matmul(kg_imag,e_real)+np.matmul(kg_real,e_imag)

        echohat_real=np.matmul(xt_real,h_posterior_real).squeeze().squeeze()-np.matmul(xt_imag,h_posterior_imag).squeeze().squeeze()
        echohat_imag=np.matmul(xt_real,h_posterior_imag).squeeze().squeeze()+np.matmul(xt_imag,h_posterior_real).squeeze().squeeze()

        estimated_complex=mic_real-echohat_real+1j*(mic_imag-echohat_imag)
        estimated_block = np.fft.irfft(estimated_complex)
        out_audio[idx*block_shift:idx*block_shift+block_len] +=estimated_block*windows
    return out_audio


mic_audio,sr1=sf.read('./mic_in.wav')
ref_audio,sr2=sf.read('./mic_far.wav')

out_audio=run_nkf(mic_audio,ref_audio)

sf.write('./aecout.wav',out_audio,16000)


