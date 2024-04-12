import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { TailwindcssButtons } from './button-test';

const Dropzone = () => {
    const [files, setFiles] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');

    const onDrop = useCallback(acceptedFiles => {
        setFiles(acceptedFiles);
    }, []);

    const analyzePhoto = async () => {
        if (selectedModel && files.length > 0) {

            const formData = new FormData();
            formData.append('photo', files[0]);
            formData.append('model', selectedModel);

            try {
                const response = await fetch('http://127.0.0.1:5000/analyze_photo', {
                    method: 'POST',
                    body: formData,
                });

            } catch (error) {
                
            } finally {
                
            }
        }
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

    return (
        <div>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                <option value="">Seleziona un modello</option>
                <option value="modello1">Modello 1</option>
            </select>

            <div {...getRootProps()} style={dropzoneStyle}>
                <input {...getInputProps()} />
                {isDragActive ? (
                    <p>Drop the files here ...</p>
                ) : (
                    <p>Trascina qui i file oppure clicca per selezionare</p>
                )}
                <ul style={fileListStyle}>
                    {files.map((file, index) => (
                        <li key={index} style={fileItemStyle}>{file.name}</li>
                    ))}
                </ul>
            </div>

            
            <TailwindcssButtons analyzePhoto={analyzePhoto}></TailwindcssButtons>
        </div>
    );
};


const dropzoneStyle = {
    border: '2px dashed #ccc',
    borderRadius: '10px',
    padding: '80px',
    textAlign: 'center',
    cursor: 'pointer',
    marginBottom: '20px',
    marginTop: '60px',
};

const fileListStyle = {
    listStyle: 'none',
    margin: '10px auto',
    padding: 0,
    maxWidth: '300px',
};

const fileItemStyle = {
    backgroundColor: '#f0f0f0',
    padding: '5px 10px',
    borderRadius: '4px',
    margin: '6px 0',
};

export default Dropzone;
