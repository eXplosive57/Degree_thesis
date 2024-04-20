import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { TailwindcssButtons } from './button-test';
import PropagateLoader from "react-spinners/PropagateLoader";
import { fetchPhotoList } from './page';


const Dropzone = () => {
    const [files, setFiles] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [loading, setLoading] = useState(false);

    const onDrop = useCallback(acceptedFiles => {
        setFiles(acceptedFiles);
    }, []);

    const analyzePhoto = async () => {
        if (selectedModel && files.length > 0) {
            setLoading(true);

            const formData = new FormData();
            formData.append('photo', files[0]);
            formData.append('model', selectedModel);

            try {
                const response = await fetch('http://127.0.0.1:5000/analyze', {
                    method: 'POST',
                    body: formData,
                });




                setLoading(false);  // Imposta lo stato di caricamento su false dopo che l'analisi è completa
            } catch (error) {
                console.error('Errore durante l\'analisi della foto:', error);
                setLoading(false); // Assicurati che lo stato di caricamento sia impostato su false in caso di errore
            }
        }
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

    return (
        <div>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                <option value="">Seleziona un modello</option>
                <option value="video">VIDEO DETECTION</option>
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

            {loading ? <div style={loaderContainerStyle}>
                <PropagateLoader
                    size={20}
                    color={"black"}
                    loading={loading}
                    speedMultiplier={1.5}
                    aria-label="Loading Spinner"
                    data-testid="loader"
                />
            </div> : null} {/* Mostra l'animazione di caricamento se loading è true */}

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

const loaderContainerStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: '30px',
    marginBottom: '50px'
};

export default Dropzone;
