import { readFile, readdir, writeFile } from "fs/promises";

import dotenv from "dotenv";
import { getEmbedding } from ".";
import {upsert} from "./pinecone";

dotenv.config();



const main = async () => { 
    
    // get the first and second arg from the command line
    const fileName = process.argv[2];

    // read the file into contents
    const content = await readFile(fileName, "utf8");

    const embedding = await getEmbedding(content)


    const res = await upsert(fileName, embedding);

    console.log(res)





 }