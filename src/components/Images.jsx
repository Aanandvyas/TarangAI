import React from 'react'

const Images = () => {
    return (
        <div className='w-full h-[70vh] bg-white flex items-center  justify-center '>
            <div className='w-[40%] h-1/2 relative'>
                <div className='absolute w-18 h-[6rem] -right-[40%] top-6  bg-red-500'>
                    <img className='w-full h-full object-cover' src="src\assets\img\photo2.jpg" alt="" />
                </div>
                <div className='absolute w-25 h-[4rem] bg-blue-500 aspect-video -left-1/2 top-1/3  '>
                    <video loop autoPlay muted className="w-full h-full object-cover" src='src\assets\video\video4.mp4'></video>
                </div>
                <div className='absolute w-30 h-[4.5rem] aspect-video -left-[70%] top-5/6  bg-amber-400  '>
                    <img className='w-full h-full object-cover' src="src\assets\img\coverphoto13.jpg" alt="" />
                </div>
                <div className='absolute w-27 h-[7rem] bg-amber-950 top-[83%] -right-[65%]'>
                    <img className='w-full h-full object-cover' src="src\assets\img\coverphoto6.webp" alt="" />
                </div>
                <img className='w-full h-full object-cover' src="src\assets\img\coverphoto9.jpg" alt="" />
            </div>
        </div>
    )
}

export default Images