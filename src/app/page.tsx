"use client";
import React, { useState, useEffect } from 'react';
import { ThreeDCardDemo } from "./3d-test";
import { SparklesPreview } from "./spark-test";
import './globals.css'
import { TabsDemo } from './tabs-test';
import socketIOClient from 'socket.io-client';
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell } from "@nextui-org/react";
import download from './icon/download.png';
import green_icon from './icon/green_icon.png';
import red_icon from './icon/red_icon.png';
import analyze from './icon/analyze.png';
import sapienza from './icon/sap_logo.jpg';

/* need to declare types of data received from server */
interface result {
  video: string;
  anteprima: string;
  file_name: string;
  state: string
}


export default function Home() {
  const [photoList, setPhotoList] = useState<result[]>([]);


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
                        element.state == 'Real' ?(
                          <div style={{ display: 'flex', alignItems: 'center' }}>
                            
                            {element.state}   <img src={green_icon.src} alt="boh" style={{ maxWidth: '25px', maxHeight: '25px', marginLeft: '5px' }} />
                              
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center' }}>
                              {element.state} <img src={red_icon.src} alt="boh" style={{ maxWidth: '25px', maxHeight: '25px', marginLeft: '5px' }} />
                              
                          </div>
                        )
                    )
                    :
                    (

                      <div style={{ display: 'flex', alignItems: 'center' }}>
                          <span>Analyzed</span> <img src={analyze.src} alt="boh" style={{ maxWidth: '25px', maxHeight: '25px', marginLeft: '5px' }} />
                      </div>
                      
                    )
                                       
                  }</TableCell>
                  {/* check if the element is a video or not */}
                  <TableCell>
                    {element.video ? (
                      <div>
                        <a href={`data:video/mp4;base64,${element.video}`} download={`${element.file_name}.mp4`}><img src={download.src}/></a>
                      </div>
                    ) : (
                      <div>
                        <a href={element.anteprima} download><img src={download.src} alt=''/></a>
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

      <footer className="footer">
    <div>
      <p>© 2024 Kerrouche Ilyas - Ingegneria Dell'informazione</p> <img src={sapienza.src} alt="Logo" style={{ maxWidth: '200px', maxHeight: '200px' }} />
    </div>
  </footer>
    </main >


  );
}