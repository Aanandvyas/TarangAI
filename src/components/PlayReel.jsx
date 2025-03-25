import React, { useEffect, useRef } from "react";
import videoFile from "../assets/video/playreel1.mp4"; // Ensure correct path
import gsap, { ScrollTrigger, Power4 } from "gsap/all";

const PlayReel = () => {
    const parent = useRef(null);
    const videoContainer = useRef(null);
    const play = useRef(null);
    const reel = useRef(null);

    useEffect(() => {
        gsap.registerPlugin(ScrollTrigger);
        var t1 = gsap.timeline({
            scrollTrigger: {
                trigger: parent.current,
                start: "top top",
                pin: true,
                scrub: 1.5,
                // markers: true // Set to false or remove when not debugging

            },
        })
            t1.to(videoContainer.current, {

                scale: 1.7,
                ease: Power4.easeOut,
            },'a')
            .to(play.current,{
                x:"-100%",
                scale:1,
                ease: Power4
            },'a')
            .to(reel.current,{
                x:"100%",
                scale:1,
                ease: Power4
            },'a')
    });

    return (
        <div
            ref={parent}
            className="w-full h-screen overflow-hidden flex flex-col text-white items-center justify-center bg-black font-mono relative"
        >
            <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white px-1 md:px-16 z-10 flex gap-10 md:gap-20">
                <h1 ref={play} className="sm:text-8xl text-4xl md:text-9xl font-light leading-none">
                    Play
                </h1>
                <h1 ref={reel} className="sm:text-8xl text-4xl md:text-9xl font-light leading-none">
                    Reel
                </h1>
            </div>
            <div
                ref={videoContainer}
                className="relative w-full max-w-[80vw] md:max-w-[60vw] lg:max-w-[50vw] aspect-video overflow-hidden"
            >
                <video
                    autoPlay
                    loop
                    muted
                    className="w-full h-full object-cover"
                    src={videoFile}
                ></video>
            </div>
            
        </div>
    );
};

export default PlayReel;
