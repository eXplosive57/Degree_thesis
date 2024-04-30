"use client";
import React, { useState, useEffect } from 'react';
import { ThreeDCardDemo } from "./3d-test";
import { SparklesPreview } from "./spark-test";
import './globals.css'
import { TabsDemo } from './tabs-test';
import socketIOClient from 'socket.io-client';
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell, RadioGroup, Radio } from "@nextui-org/react";


/* need to declare types of data received from server */
interface result {
  video: string;
  frame: string;
  name: string;
  anteprima: string;
  file_name: string
}


export default function Home() {
  const [photoList, setPhotoList] = useState<result[]>([]);


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

    <main className="dark text-foreground bg-background">

      <SparklesPreview />
      <TabsDemo />



      <div className="content">
        <div className="container">
          <Table
            color={'primary'}
            selectionMode="multiple"
            aria-label="Example static collection table"
          >
            <TableHeader>
              <TableColumn>IMAGE</TableColumn>
              <TableColumn>NAME</TableColumn>
              <TableColumn>ACTION</TableColumn>

            </TableHeader>
            <TableBody>
              {photoList.map((photo, index) => ( // Utilizza photo come elemento corrente
                <TableRow key={index}>

                  {/* ignore errors on imageUrl and name */}
                  <TableCell>
                    <img src= {photo.anteprima} alt={`Photo ${index}`} style={{ maxWidth: '80px' }} />
                  </TableCell>
                  <TableCell>{photo.file_name}</TableCell>
                  <TableCell>download</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>


        </div>
      </div>







    </main >


  );
}