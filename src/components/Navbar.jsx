import React, { useState } from "react";
import { motion } from "framer-motion";
import { IoMenuSharp } from "react-icons/io5";
import { LuSquareArrowOutUpRight } from "react-icons/lu";
import { X } from "lucide-react";

const Navbar = () => {
    const [menuOpen, setMenuOpen] = useState(false);

    return (
        <div className="w-full fixed top-0 z-50 bg-black bg-opacity-80 backdrop-blur-md py-4 px-6">
            <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-white">
                {/* Logo */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 1 }}
                    className="text-2xl font-bold tracking-wide"
                >
                    TarangAI
                </motion.div>

                {/* Desktop Button */}
                <motion.div
                    className="hidden sm:flex"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <a
                        href="/tarang"
                        className="px-6 py-2 bg-white text-black text-lg font-medium rounded-full shadow-md transition-transform transform hover:scale-105 flex items-center gap-2"
                    >
                        Try Tarang <LuSquareArrowOutUpRight />
                    </a>
                </motion.div>

                {/* Mobile Menu Icon */}
                <motion.div
                    className="sm:hidden text-3xl cursor-pointer"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    onClick={() => setMenuOpen(!menuOpen)}
                >
                    {menuOpen ? <X /> : <IoMenuSharp />}
                </motion.div>
            </div>

            {/* Mobile Menu */}
            {menuOpen && (
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="sm:hidden absolute top-16 left-0 w-full bg-black bg-opacity-90 backdrop-blur-md p-6 flex flex-col items-center"
                >
                    <a
                        href="/tarang"
                        className="px-6 py-2 bg-white text-black text-lg font-medium rounded-full shadow-md transition-transform transform hover:scale-105 flex items-center gap-2"
                    >
                        Try Tarang <LuSquareArrowOutUpRight />
                    </a>
                </motion.div>
            )}
        </div>
    );
};

export default Navbar;