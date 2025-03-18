import React from "react";

const AboutPage = () => {
    return (
        <div className="min-h-screen bg-[#070707] text-[#E0CCBB] flex flex-col items-center justify-center px-6 md:px-20">
            <div className="max-w-4xl w-full">
                <div className="w-full flex justify-between items-center">
                    <h1 className="text-5xl md:text-7xl font-light leading-tight text-left">
                        Our <br />
                        Story
                    </h1>
                    <video muted autoPlay loop className="w-1/2 h-auto object-contain  ml-4" src="src\assets\video\video5.mp4"></video>
                </div>
                <p className="mt-6 text-lg ">
                    Lorem ipsum dolor sit amet consectetur adipisicing elit. Perferendis?
                </p>
                <a
                    href="#"
                    className="mt-6 inline-block text-lg  underline  hover:opacity-80 transition"
                >
                    Anand Vyas - Founder
                </a>
                <div className="w-full h-px bg-[#E0CCBB] my-8"></div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6  text-sm">
                    <a href="#" className="hover:text-white transition">Work</a>
                    <a href="#" className="hover:text-white transition">Github</a>
                    <a href="#" className="hover:text-white transition">Studio</a>
                    <a href="#" className="hover:text-white transition">Dribbble</a>
                    <a href="#" className="hover:text-white transition">News</a>
                    <a href="#" className="hover:text-white transition">LinkedIn</a>
                    <a href="#" className="hover:text-white transition">Contact</a>
                    <a href="#" className="hover:text-white transition">Instagram</a>
                </div>
            </div>
        </div>
    );
};

export default AboutPage;
