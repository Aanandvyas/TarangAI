<<<<<<< HEAD
import { div, video } from 'framer-motion/client'
=======
import { div } from 'framer-motion/client'
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
import React, { useState } from 'react'

const Nritya = () => {

<<<<<<< HEAD
    const [elems,setElems] = useState([{
        heading:"Kathak",
        subheading:"Kathak is one of the eight forms of Indian classical dance. This dance form traces its origins to the nomadic bards of ancient northern India, known as Kathakars or storytellers. Kathak is characterized by intricate footwork, spins, and expressive gestures. The dance form has evolved over the centuries, blending Hindu and Muslim traditions, and is known for its grace, elegance, and storytelling.",
        video:"/src/assets/video/video1.mp4",
        img:"/src/assets/img/photo1.jpg"
=======
    useState([{
        heading:"Kathak",
        subheading:"Kathak is one of the eight forms of Indian classical dance. This dance form traces its origins to the nomadic bards of ancient northern India, known as Kathakars or storytellers. Kathak is characterized by intricate footwork, spins, and expressive gestures. The dance form has evolved over the centuries, blending Hindu and Muslim traditions, and is known for its grace, elegance, and storytelling.",
        video:"src\assets\video\video1.mp4",
        img:"src\assets\img\photo1.jpg"
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
    },
    {
        heading:"Bharatanatyam",
        subheading:"Bharatanatyam is one of the oldest and most popular forms of Indian classical dance. Originating in Tamil Nadu, Bharatanatyam is known for its grace, purity, and sculptural poses. The dance form is characterized by intricate footwork, hand gestures, and facial expressions, and is performed to classical Carnatic music. Bharatanatyam is a vibrant expression of India’s rich cultural heritage and mythology.",
<<<<<<< HEAD
        video:"/src/assets/video/video2.mp4",
        img:"/src/assets/img/photo2.jpg"
=======
        video:"src\assets\video\video2.mp4",
        img:"src\assets\img\photo2.jpg"
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
    },
    {
        heading:"Kuchipudi",
        subheading:"Kuchipudi is a classical dance form that originated in the village of Kuchipudi in Andhra Pradesh. This dance form combines elements of dance, drama, and music, and is known for its graceful movements, fast footwork, and expressive storytelling. Kuchipudi performances often include themes from Hindu mythology and are accompanied by Carnatic music. The dance form has gained international recognition for its beauty and cultural significance.",
<<<<<<< HEAD
        video:"/src/assets/video/video3.mp4",
        img:"/src/assets/img/photo3.jpg"
=======
        video:"src\assets\video\video3.mp4",
        img:"src\assets\img\photo3.jpg"
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
    },
    {
        heading:"Odissi",
        subheading:"Odissi is one of the oldest forms of Indian classical dance, originating in the temples of Odisha. This dance form is known for its fluid movements, intricate footwork, and expressive storytelling. Odissi performances often depict stories from Hindu mythology and are accompanied by classical Odissi music. The dance form is characterized by its graceful poses, sculpturesque postures, and emotive expressions.",
<<<<<<< HEAD
        video:"/src/assets/video/video4.mp4",
        img:"/src/assets/img/photo4.jpg"
=======
        video:"src\assets\video\video4.mp4",
        img:"src\assets\img\photo4.jpg"
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
    }
])

    return (
        <div className='w-full relative'>
<<<<<<< HEAD
            <div className='py-14 mx-auto max-w-screen-2xl sm:px-9 px-9'>
                <h1 className='text-6xl font-mono sm:text-[10rem] sm:leading-none sm:tracking-tight '>Nritya</h1>
                <p className='mt-6 font-mono text-md sm:text-xl sm:leading-8 leading-5'>This part offers a glimpse into the diverse traditional dances of India, each reflecting the rich history and culture of its state. Dance is not just an art form but a vibrant expression of India’s heritage, captivating the world with its beauty, storytelling, and deep cultural significance.</p>
                <div className='elems sm:flex sm:flex-wrap sm:gap-8 mt-9 '>
                    {elems.map((item,index) => {
                        return <div key={index} className='elem w-full sm:w-[48%] mt-10 ' >
                        <div className='video w-full h-[105vw] sm:h-96 relative overflow-hidden  '>
                            <img className='hidden sm:block object-cover h-full w-full' src={item.img} alt="" />
                            <video autoPlay muted loop className='block sm:hidden scale-[1.5] h-full w-full absolute ' src={item.video}></video>
=======
            <div className='py-20 mx-auto max-w-screen-2xl sm:px-10 px-9'>
                <h1 className='text-6xl font-mono'>Nritya</h1>
                <p className='mt-6 font-mono text-md sm:text-xl sm:leading-8 leading-5'>This part offers a glimpse into the diverse traditional dances of India, each reflecting the rich history and culture of its state. Dance is not just an art form but a vibrant expression of India’s heritage, captivating the world with its beauty, storytelling, and deep cultural significance.</p>
                <div className='elems mt-9 '>
                    <div className='elem w-full ' >
                        <div className='video w-full h-[105vw] relative overflow-hidden  '>
                            <img className='hidden sm:block object-cover h-full w-full' src="src\assets\img\photo1.jpg" alt="" />
                            <video autoPlay muted loop className='block sm:hidden scale-[1.5] h-full w-full absolute ' src="src\assets\video\video1.mp4"></video>
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
                        </div>
                        <div className='mt-4'>
                            <h3 className='font-semibold'>Lorem, ipsum dolor.</h3>
                            <h3 className='opacity-40'>Lorem ipsum dolor consectetur adipisicing elit. Possimus porro distinctio veniam!</h3>
                        </div>
                    </div>
<<<<<<< HEAD
                    })}
=======
>>>>>>> 0c63791639347acf56a8b9c1bf42574fe060abbc
                </div>
            </div>
        </div>
    )
}

export default Nritya