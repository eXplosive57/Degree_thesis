"use client";
import React, { useState, useEffect } from 'react';
import { ThreeDCardDemo } from "./3d-test";
import { SparklesPreview } from "./spark-test";
import './css-tabella/css/style.css';
import './css-tabella/css/bootstrap.min.css';
import './css-tabella/fonts/icomoon/style.css';
import './style.css'
import { TabsDemo } from './tabs-test';
import { TailwindcssButtons } from './button-test';

export default function Home() {
  const [photoList, setPhotoList] = useState([]);

  // Estraggo le foto analizzate salvate nella directory lato server
  useEffect(() => {
    const fetchPhotoList = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/photo_list');
        const data = await response.json();
        setPhotoList(data);
      } catch (error) {
        console.error('Errore durante il recupero dell\'elenco delle foto:', error);
      }
    };

    fetchPhotoList();
  }, [])

  return (
    <main>
      <SparklesPreview />
      <TabsDemo />
      <div className="content">
        <div className="container">
          <h2 className="mb-5">Download Results</h2>
          <div className="table-responsive">
            <table className="table table-striped custom-table">
              <thead>
                <tr>
                  <th scope="col">File Name</th>
                </tr>
              </thead>
              <tbody>
                {photoList.map((imageUrl, index) => (
                  <tr key={index} scope="row">
                    <td><img src={imageUrl} alt={`Photo ${index}`} style={{ maxWidth: '500px' }} /></td>
                    <td>prova</td>
                    <td><small className="d-block">Far far away, behind the word mountains</small></td>
                    <td><a href={imageUrl} download>Download</a></td>
                    <td><button onClick={() => handleDelete(index)}>Delete</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}
