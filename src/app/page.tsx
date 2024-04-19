"use client";
import React, { useState, useEffect } from 'react';
import { ThreeDCardDemo } from "./3d-test";
import { SparklesPreview } from "./spark-test";
import './globals.css'
import { TabsDemo } from './tabs-test';
import socketIOClient from 'socket.io-client';
import { Card, CardHeader, CardBody, CardFooter, Divider, Link, Image } from "@nextui-org/react";
import { NextUIProvider } from "@nextui-org/react";



export default function Home() {
  const [photoList, setPhotoList] = useState([]);


  // Estraggo le foto analizzate salvate nella directory lato server
  async function fetchPhotoList() {
    try {
      const response = await fetch('http://127.0.0.1:5000/photo_list');
      const data = await response.json();
      setPhotoList(data);
    } catch (error) {
      console.error('Errore durante il recupero dell\'elenco delle foto:', error);
      return [];
    }
  }

  useEffect(() => {
    const socket = socketIOClient('http://127.0.0.1:5000')

    socket.on('photo_analyzed_notification', () => {
      // quando ricevo una notifica dal client, aggiorno lista foto

      // -------------- PROVA A METTERE QUI IL LOADING ANIMAZIONE --------------
      fetchPhotoList();
    });
  }, []);

  useEffect(() => {
    fetchPhotoList();
  }, []);// Le parentesi quadre vuote indicano che questo effetto non dipende da nessuna variabile e quindi verrà eseguito solo una volta dopo il montaggio del componente.



  return (

    <main>
      <SparklesPreview />
      <TabsDemo />

      <NextUIProvider>

        <Card className="py-4">
          <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
            <p className="text-tiny uppercase font-bold">Daily Mix</p>
            <small className="text-default-500">12 Tracks</small>
            <h4 className="font-bold text-large">Frontend Radio</h4>
          </CardHeader>
          <CardBody className="overflow-visible py-2">
            <Image
              alt="Card background"
              className="object-cover rounded-xl"
              src="https://avatars.githubusercontent.com/u/86160567?s=200&v=4"
              width={270}
            />
          </CardBody>
        </Card>

      </NextUIProvider>


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
                  <tr key={index}>
                    <td><img src={imageUrl} alt={`Photo ${index}`} style={{ maxWidth: '500px' }} /></td>
                    <td>prova</td>
                    <td><small className="d-block">Far far away, behind the word mountains</small></td>
                    <td><a href={imageUrl} download>Download</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>











    </main >


  );
}