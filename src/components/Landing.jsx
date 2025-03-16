import React from 'react'

const Landing = () => {
    return (
        <div className='relative w-full h-[200vh] '>
            <div className='picture w-full h-full '>
                <img className='w-full h-full object-cover' src="src\assets\img\coverphoto13.jpg" alt="" />
            </div>
            <div className='absolute top-0 h-full  text-white max-w-screen-2xl mx-auto px-5 sm:px-10'>
                <div className='font-medium text-white text-xl mt-65 ml-1'>
                    <p>The roots of Indian dance forms lie in Lord Shiva’s </p>
                    <p>Nataraja manifestation.Bharatanatyam, Kathakali, Odissi, Kathak,</p>
                    <p> and Kuchipudi evolved from his Tandava and Lasya dance, </p>
                    <p> symbolizing devotion, beauty, and spirituality.</p>
                </div>
            </div>
        </div>
    )
}

export default Landing