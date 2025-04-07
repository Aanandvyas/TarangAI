import React from 'react'
import { LuSquareArrowOutUpRight } from "react-icons/lu";
import { motion } from "framer-motion";
import { Power4 } from 'gsap';
import { i } from 'framer-motion/client';


const Landing = () => {
    return (
        <div className='relative w-full h-[150vh] sm:h-[250vh] '> 
            <div className='picture w-full h-full overflow-hidden '>
                <img data-scroll data-scroll-speed="-1" className='w-full h-full object-cover ' src="src\assets\img\coverphoto14.jpg" alt="" />
            </div>
            <div className='absolute w-full top-0 h-full text-white sm:ml-15 max-w-screen-2xl mx-auto px-5 sm:px-10'>
                <div className='font-medium text-white text-md sm:text-xl mt-45 ml-1 sm:mt-65'>
                    {["The roots of Indian classical dance can be traced back to Lord Shiva’s", "Nataraja manifestation. Bharatanatyam, Kathakali, Odissi, Kathak,", "and Kuchipudi evolved from his Tandava and Lasya dance,", "symbolizing devotion, beauty, and spirituality."]
                        .map((item, index) => {
                            return (
                                <p className='text-md font-regular  overflow-hidden  sm:text-2xl leading-8'>
                                    <motion.span
                                        initial={{ rotate: 90, y: "100%", opacity: 0 }}
                                        animate={{ rotate: 0, y: 0, opacity: 1 }}
                                        transition={{ ease: [0.22, 1, 0.36, 1], duration: 1, delay: index * 0.2 }}
                                        className='inline-block origin-left'
                                    >
                                        {item}

                                    </motion.span>
                                </p>
                            )

                        })}

                </div>
                <div className='heading mt-10'>
                    {["Learn", "Design", "Generate"].map((item, index) => (
                        <h1 key={index} className='sm:text-[9rem]   text-3xl sm:py-6 py-2 leading-none tracking-tighter overflow-hidden'>
                            <motion.span
                                className='inline-block origin-left'
                                initial={{ rotate: 90, y: "100%", opacity: 0 }}
                                animate={{ rotate: 0, y: 0, opacity: 1 }}
                                transition={{ ease: [0.22, 1, 0.36, 1], duration: 1, delay: index * 0.4 }}
                            >
                                {item}
                            </motion.span>
                        </h1>
                    ))}
                </div>

                <div className='para2  text-md font-regular  font-medium text-white sm:text-xl mt-12 sm:mt-20 ml-1 '>

                    <p>"We bring India’s rich cultural heritage to life through </p>
                    <p>
                        AI, transforming traditional dance poses and art styles
                    </p>
                    <p>into stunning digital experiences. Immerse yourself </p>
                    <p className='mb-5'>in the beauty of tradition with just a click."</p>
                    <div className='sm:text-xl flex items-center gap-3 mt-10'>
                        {/* Animated Button */}
                        <motion.a
                            href="#"
                            className='border-b-[.3px] inline-flex border-zinc-100 pb-1 text-lg'
                            whileHover={{ scale: 1.05, y: -2 }}
                            transition={{ duration: 0.3, ease: "easeInOut" }}
                        >
                            Let's try
                        </motion.a>

                        {/* Animated Arrow Icon */}
                        <motion.p
                            className='w-6 h-9 pt-2'
                            whileHover={{ x: 5, opacity: 1 }}
                            initial={{ opacity: 0.6 }}
                            transition={{ duration: 0.3, ease: "easeInOut" }}
                        >
                            <LuSquareArrowOutUpRight />
                        </motion.p>
                    </div>

                </div>
            </div>
        </div>
    )
}

export default Landing 