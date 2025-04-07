import React from 'react';
import { ArrowRight } from "lucide-react";
import { motion } from "framer-motion";


const Generate = () => {
    return (
        <div className="w-full bg-white py-20">
            <div className="max-w-screen-2xl mx-auto px-5 sm:px-10">
                <div className="flex items-center justify-center">
                    <div className="text-center">
                        {["Generate", "Dance Forms"].map((item, index) => (
                            <h1 
                                key={index}
                                className="capitalize text-5xl sm:text-9xl font-extrabold tracking-tight font-serif overflow-hidden"
                            >
                                <motion.span 
                                    initial={{ rotate: 90, y: "40%", opacity: 0 }}
                                    whileInView={{ rotate: 0, y: 0, opacity: 1 }}
                                    viewport={{ once: true }}
                                    className="inline-block origin-left"
                                    transition={{ ease: [0.22, 1, 0.36, 1], duration: 1, delay: 0.2 }}
                                >
                                    {item}
                                </motion.span>
                            </h1>
                        ))}
                        
                        {/* Description */}
                        <p className="w-4/5 sm:w-1/3 mx-auto mt-6 sm:mt-10 sm:text-xl text-base leading-relaxed text-gray-700">
                            Experience the power of AI in generating traditional dance forms with precision and accuracy.
                        </p>

                        {/* Browse Button */}
                        <a
                            className="relative border-b-[1px] border-zinc-900 pb-2 mt-8 inline-flex items-center gap-2 transition-all duration-300 hover:bg-black hover:text-white hover:px-2 group"
                            href="#"
                        >
                            Browse all new
                            <span className="opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300">
                                <ArrowRight className="w-5 h-5" />
                            </span>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Generate;
