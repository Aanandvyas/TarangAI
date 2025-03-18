import React from "react";
import videoFile from "../assets/video/playreel.mp4"; // Ensure correct path

const PlayReel = () => {
    return (
        <div className="w-full h-screen overflow-hidden flex flex-col text-white items-center justify-center bg-black font-mono relative">
            <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white px-1 md:px-16 z-10 flex gap-10 md:gap-20">
                <h1 className="sm:text-8xl text-4xl md:text-9xl font-light leading-none">
                    Play
                </h1>
                <h1 className="sm:text-8xl text-4xl md:text-9xl font-light leading-none">
                    Reel
                </h1>
            </div>
            <div className="relative w-full max-w-[80vw] md:max-w-[60vw] lg:max-w-[50vw] aspect-video overflow-hidden">
                <video 
                    autoPlay 
                    loop 
                    muted 
                    className="w-full h-full object-cover scale-[1.2]"
                    src={videoFile}
                ></video>
            </div>
            <p className="absolute bottom-10 text-center px-6 text-xs md:text-sm">
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Consequuntur.
            </p>
        </div>
    );
}

export default PlayReel;
