"use client";

import Image from "next/image";
import { Tabs } from "./components/ui/tabs";
import Dropzone from './Dropzone';


export function TabsDemo() {
    const tabs = [
        {
            title: "",
            value: "",
            content: (
                <div className="w-full relative h-auto rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-blue-500 to-violet-900 ">
                    <p>SELECT MODEL</p>
                    {/* <h1>DRAG AND DROP FOTO</h1> */}
                    <Dropzone />
                </div>
            ),
        },
        // {
        //     title: "VIDEO",
        //     value: "VIDEO",
        //     content: (
        //         <div className="w-full overflow-hidden relative h-auto rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-blue-500 to-purple-800">
        //             <p>VIDEO</p>
        //             <h1>DRAG AND DROP VIDEO</h1>
        //             <Dropzone />
        //         </div>
        //     ),
        // },
        // {
        //     title: "Random",
        //     value: "random",
        //     content: (
        //         <div className="w-full overflow-hidden relative h-auto rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-blue-700 to-violet-900">
        //             <p>Random tab</p>
        //             <p>?</p>
        //         </div>
        //     ),
        // },
    ];

    return (
        <div className="h-[20rem] md:h-[40rem] [perspective:1000px] relative b flex flex-col max-w-5xl mx-auto w-full  items-start justify-start" style={{ marginTop: '-10rem', marginBottom: '15rem' }
        }>
            <Tabs tabs={tabs} />
        </div >
    );
}
