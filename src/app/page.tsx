"use client";
import React, { useState, useEffect } from 'react';
import { ThreeDCardDemo } from "./3d-test";
import { SparklesPreview } from "./spark-test";
import './globals.css'
import { TabsDemo } from './tabs-test';
import socketIOClient from 'socket.io-client';
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell } from "@nextui-org/react";
import arrow from './icon/arrow2.png';

/* need to declare types of data received from server */
interface result {
  video: string;
  anteprima: string;
  file_name: string;
  state: string
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
          >
            <TableHeader>
              <TableColumn>IMAGE</TableColumn>
              <TableColumn>NAME</TableColumn>
              <TableColumn>STATE</TableColumn>
              <TableColumn>ACTION</TableColumn>
              
            </TableHeader>
            <TableBody>
              {photoList.map((element, index) => ( // Utilizza photo come elemento corrente
                <TableRow key={index}>
                  <TableCell>
                    <img src={element.anteprima} alt={`Photo ${index}`} style={{ maxWidth: '90px', borderRadius: '6px' }} />
                  </TableCell>
                  <TableCell>{element.file_name}</TableCell>
                  <TableCell>
                    {element.state ? (
                        element.state
                    )
                    :
                    (
                      <span>Analyzed</span>
                    )
                    
                    
                  }</TableCell>
                  {/* check if the element is a video or not */}
                  <TableCell>
                    {element.video ? (
                      <div>
                        <a href={`data:video/mp4;base64,${element.video}`} download={`${element.file_name}.mp4`}><img src={arrow.src}/></a>
                      </div>
                    ) : (
                      <div>
                        <a href={element.anteprima} download><img src={arrow.src} alt=''/></a>
                      </div>
                    )
                    
                  }
                  </TableCell>

                </TableRow>


              ))}
            </TableBody>
          </Table>


        </div>
      </div>







    </main >


  );
}