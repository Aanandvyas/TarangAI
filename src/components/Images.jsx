import gsap, { Linear, Power4, ScrollTrigger } from 'gsap/all';
import React, { useEffect, useRef } from 'react';

const Images = () => {
    const parent = useRef(null);
    const first = useRef(null);
    const second = useRef(null);
    const third = useRef(null);
    const fourth = useRef(null);

    useEffect(() => {
        gsap.registerPlugin(ScrollTrigger);
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: parent.current,
                start: "0 90%",
                scrub: 1,
                // markers: true, // Uncomment for debugging
            }
        });

        tl
            .to(first.current, {
                x: "45%",
                ease: Linear
            })
            .to(second.current, {
                x: "-30%",
                ease: Linear
            })
            .to(third.current, {
                x: "-50%",
                ease: Linear
            })
            .to(fourth.current, {
                x: "40%",
                ease: Linear
            });
    }, []);

    return (
        <div ref={parent} className="w-full h-[100vh] bg-white flex items-center justify-center">
            <div className="w-[40%] sm:w-[20%] sm:h-[80%] h-1/2 relative">
                <div ref={first} className="absolute w-20 h-[8rem] -right-[40%] top-6 sm:w-50 sm:h-[18rem]">
                    <img className="w-full h-full object-cover" src="src/assets/img/photo2.jpg" alt="" />
                </div>
                <div ref={second} className="absolute w-25 h-[6rem] sm:w-[17rem] sm:h-[18rem] bg-blue-500 aspect-video -left-1/2 sm:-left-2/3 top-1/3 sm:top-1/6">
                    <video loop autoPlay muted className="w-full h-full object-cover" src="src/assets/video/video4.mp4"></video>
                </div>
                <div ref={third} className="absolute w-30 h-[4rem] sm:w-[15rem] sm:h-[9rem] aspect-video -left-[70%] sm:-left-[50%] top-5/6 bg-amber-400">
                    <img className="w-full h-full object-cover" src="src/assets/img/coverphoto13.jpg" alt="" />
                </div>
                <div ref={fourth} className="absolute w-27 h-[5rem] sm:w-[15rem] sm:h-[8rem] bg-amber-950 top-[89%] -right-[65%] sm:-right-[65%]">
                    <video loop autoPlay muted className="w-full h-full object-cover" src="src/assets/video/playreel.mp4"></video>
                </div>
                <img className="w-full h-full object-cover" src="src/assets/img/coverphoto9.jpg" alt="" />
            </div>
        </div>
    );
};

export default Images;
