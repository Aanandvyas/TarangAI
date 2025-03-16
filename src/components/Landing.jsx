import React from 'react'
import { LuSquareArrowOutUpRight } from "react-icons/lu";


const Landing = () => {
    return (
        <div className='relative w-full h-[150vh] sm:h-[250vh] '>
            <div className='picture w-full h-full '>
                <img className='w-full h-full object-cover ' src="src\assets\img\coverphoto14.jpg" alt="" />
            </div>
            <div className='absolute w-full top-0 h-full text-white sm:ml-15 max-w-screen-2xl mx-auto px-5 sm:px-10'>
                <div className='font-medium text-white text-md sm:text-xl mt-45 ml-1 sm:mt-65'>
                    <p className='text-md font-regular  sm:text-2xl leading-8'>The roots of Indian dance forms lie in Lord Shiva’s </p>
                    <p className='text-md font-regular  sm:text-2xl leading-8'>Nataraja manifestation. Bharatanatyam, Kathakali, Odissi, Kathak,</p>
                    <p className='text-md font-regular  sm:text-2xl leading-8'> and Kuchipudi evolved from his Tandava and Lasya dance, </p>
                    <p className='text-md font-regular  sm:text-2xl leading-8'> symbolizing devotion, beauty, and spirituality.</p>
                </div>
                <div className='heading mt-16 sm:text-[9rem] font-light text-6xl leading-none tracking-tighter ' >
                    <h1>Generate</h1>
                    <h1>Learn</h1>
                    <h1>Improve</h1>
                </div>
                <div className='para2  text-md font-regular  font-medium text-white sm:text-xl mt-12 sm:mt-17 ml-1 '>

                    <p>"We bring India’s rich cultural heritage to life through </p>
                    <p>
                        AI, transforming traditional dance poses and art styles
                    </p>
                    <p>into stunning digital experiences. Immerse yourself </p>
                    <p className='mb-5'>in the beauty of tradition with just a click."</p>
                    <div className='sm:text-xl flex gap-3 mt-10 '>
                        <a className='border-b-[.3px] inline-flex border-zinc-100 pb-1 ' href="#"> Let's try  </a> 
                        <p className='w-6 h-9 pt-2 '><LuSquareArrowOutUpRight /></p>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Landing 