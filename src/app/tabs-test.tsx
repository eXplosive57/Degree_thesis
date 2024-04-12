"use client";

import Image from "next/image";
import { Tabs } from "./components/ui/tabs";
import Dropzone from './Dropzone';
import './globals.css';

export function TabsDemo() {
    const tabs = [
        {
            title: "FOTO",
            value: "FOTO",
            content: (
                <div className="w-full overflow-y-scroll relative h-full rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-orange-700 to-red-900">
                    <p>FOTO</p>
                    <h1>DRAG AND DROP FOTO</h1>
                    <Dropzone />
                </div>
            ),
        },
        {
            title: "VIDEO",
            value: "VIDEO",
            content: (
                <div className="w-full overflow-hidden relative h-full rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-green-700 to-violet-900">
                    <p>VIDEO</p>
                    <h1>DRAG AND DROP VIDEO</h1>
                    <Dropzone />

                </div>
            ),
        },
        {
            title: "Random",
            value: "random",
            content: (
                <div className="w-full overflow-hidden relative h-full rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-blue-700 to-violet-900">
                    <p>Random tab</p>
                    <p>?</p>
                </div>
            ),
        },
    ];

    return (
        <div className="h-[20rem] md:h-[40rem] [perspective:1000px] relative b flex flex-col max-w-5xl mx-auto w-full  items-start justify-start my-40">
            <Tabs tabs={tabs} />
        </div>
    );
}

const DummyContent = () => {
    return (
        <Image
            src="/linear.webp"
            alt="dummy image"
            width="1000"
            height="1000"
            className="object-cover object-left-top h-[60%]  md:h-[90%] absolute -bottom-10 inset-x-0 w-[90%] rounded-xl mx-auto"
        />
    );
};
