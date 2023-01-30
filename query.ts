import "dotenv/config";

import {query, upsert} from "./pinecone";
import { readFile, readdir, writeFile } from "fs/promises";

import { getEmbedding } from ".";

const main = async () => { 
    
    // const q = process.argv[2];
    const q = "what is drews ktt?"


    const embedding = await getEmbedding(q)


    const res = await query(embedding);

    console.log(res)





 }

 main()