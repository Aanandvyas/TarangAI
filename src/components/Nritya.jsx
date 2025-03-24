import { div } from "framer-motion/client";
import { motion } from "framer-motion";
import React, { useState } from "react";

const Nritya = () => {

  const [elems, setElems] = useState([
    {
      heading: "Kathak",
      subheading:
        "Kathak is one of the eight forms of Indian classical dance. This dance form traces its origins to the nomadic bards of ancient northern India, known as Kathakars or storytellers. Kathak is characterized by intricate footwork, spins, and expressive gestures. The dance form has evolved over the centuries, blending Hindu and Muslim traditions, and is known for its grace, elegance, and storytelling.",
      video: "src/assets/video/video1.mp4",
      img: "src/assets/img/photo1.jpg",
    },
    {
      heading: "Bharatanatyam",
      subheading:
        "Bharatanatyam is one of the oldest and most popular forms of Indian classical dance. Originating in Tamil Nadu, Bharatanatyam is known for its grace, purity, and sculptural poses. The dance form is characterized by intricate footwork, hand gestures, and facial expressions, and is performed to classical Carnatic music. Bharatanatyam is a vibrant expression of India’s rich cultural heritage and mythology.",
      video: "src/assets/video/video2.mp4",
      img: "src/assets/img/photo2.jpg",
    },
    {
      heading: "Kuchipudi",
      subheading:
        "Kuchipudi is a classical dance form that originated in the village of Kuchipudi in Andhra Pradesh. This dance form combines elements of dance, drama, and music, and is known for its graceful movements, fast footwork, and expressive storytelling. Kuchipudi performances often include themes from Hindu mythology and are accompanied by Carnatic music. The dance form has gained international recognition for its beauty and cultural significance.",
      video: "src/assets/video/video3.mp4",
      img: "src/assets/img/photo3.jpg",
    },
    {
      heading: "Odissi",
      subheading:
        "Odissi is one of the oldest forms of Indian classical dance, originating in the temples of Odisha. This dance form is known for its fluid movements, intricate footwork, and expressive storytelling. Odissi performances often depict stories from Hindu mythology and are accompanied by classical Odissi music. The dance form is characterized by its graceful poses, sculpturesque postures, and emotive expressions.",
      video: "src/assets/video/video4.mp4",
      img: "src/assets/img/photo4.jpg",
    },
  ]);

  return (
    <div className="w-full relative">
      <div className="py-14 mx-auto max-w-screen-2xl sm:px-9 px-9">
        <h1 className="text-6xl font-mono sm:text-[10rem] py-2 overflow-hidden sm:leading-none sm:tracking-tight">
          <motion.span
            initial={{ rotate: 90, y: "40%", opacity: 0 }}
            whileInView={{ rotate: 0, y: 0, opacity: 1 }}
            viewport={{ once: true }}
            className="inline-block origin-left"
            transition={{ ease: [0.22, 1, 0.36, 1], duration: 0.8, delay: 0.2 }}
          >
            Nritya
          </motion.span>

        </h1>
        <p className="mt-6 font-mono text-md sm:text-xl sm:leading-8 leading-5">
          This part offers a glimpse into the diverse traditional dances of
          India, each reflecting the rich history and culture of its state.
          Dance is not just an art form but a vibrant expression of India’s
          heritage, captivating the world with its beauty, storytelling, and
          deep cultural significance.
        </p>
        <div className="elems sm:flex sm:flex-wrap sm:gap-8 mt-9">
          {elems.map((item, index) => {
            return (
              <div key={index} className="elem w-full sm:w-[48%] mt-10">
                <div className="video relative w-full h-[105vw] sm:h-96 overflow-hidden">
                  {/* Overlay image with fade-out animation on hover */}
                  <motion.img
                    initial={{ opacity: 1 }}
                    whileHover={{ opacity: 0 }}
                    data-scroll
                    data-scroll-speed="-.1"
                    className="hidden sm:absolute sm:z-20 sm:inset-0 sm:block object-cover w-full h-full"
                    src={item.img}
                    alt={item.heading}
                    loading="lazy"
                  />

                  {/* Background video element */}
                  <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="absolute inset-0 z-10 scale-[1.5] object-cover"
                    src={item.video}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>

                <div className="mt-4">
                  <h3 className="font-semibold">{item.heading}</h3>
                  <h3 className="opacity-40">{item.subheading}</h3>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Nritya;
