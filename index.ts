#!/usr/bin/env ts-node

import  "dotenv/config";

import { Configuration, OpenAIApi } from "openai";
import { readFile, readdir, writeFile } from "fs/promises";

import { upsert } from "./pinecone";

enum OpenAiModels {
  davinci = "text-davinci-003",
  curie = "text-curie-001",
  babbage = "text-babbage-001",
  ada = "text-ada-001",
  embedding = "text-embedding-ada-002",
}

// write function to read each file in the data folder in objects with the file name and the file contents
const readNotes = async () => {
  // specify the path to the folder you want to read
  const folderPath = "./data/notes/";
  const files = await readdir(folderPath);
  const fileObjects = [];

  for await (const file of files) {
    // get the file name and file content
    const fileName = file;
    const fileContent = await readFile(folderPath + file, "utf8");

    // create an object with the file name and file content
    const fileObject = {
      fileName,
      fileContent,
    };
    fileObjects.push(fileObject);
  }

  return fileObjects;
};



const writeOutSummaries = async (
  notes: { fileName: string; fileContent: string; summary: string }[]
) => {
  const folderPath = "./data/summaries/";

  for (const note of notes) {
    console.log(`${note.fileName}\n${note.summary}\n\n`);
    await writeFile(folderPath + note.fileName, note.summary);
  }
};

export const getEmbedding = async (text: string) => {
  const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const openai = new OpenAIApi(configuration);

  const response = await openai.createEmbedding({
    model: OpenAiModels.embedding,
    input: text,
  });

  console.log(response.data.data.length);

  return response.data.data[0].embedding;
};

function cosinesim(A: number[], B: number[]): number {
  var dotproduct = 0;
  var mA = 0;
  var mB = 0;
  for (var i = 0; i < A.length; i++) {
    // here you missed the i++
    dotproduct += A[i] * B[i];
    mA += A[i] * A[i];
    mB += B[i] * B[i];
  }
  mA = Math.sqrt(mA);
  mB = Math.sqrt(mB);
  var similarity = dotproduct / (mA * mB); // here you needed extra brackets
  return similarity;
}

export async function ask(prompt: string): Promise<any> {
  const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const openai = new OpenAIApi(configuration);

  const response = await openai.createCompletion({
    model: OpenAiModels.davinci,
    prompt,
    temperature: 0.81,
    max_tokens: 256,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  });

  return response.data.choices[0].text;
}

const main = async (q: string) => {
  const notes = await readNotes();

  const embeddings: any[] = [];
  for await (const note of notes) {
    embeddings.push(await getEmbedding(note.fileContent));
  }

  const promptEmb: any[] = await getEmbedding(q);

  console.log(`promptEmb.length: ${promptEmb.length}`)

  const probs: any[] = [];
  for (const emb of embeddings) {
    probs.push(cosinesim(promptEmb, emb));
  }
  console.log(probs);

  const maxIndex = probs.indexOf(Math.max(...probs));
  const ans = await ask(
    `${notes[maxIndex].fileContent}\n\nanswer the following question with the above info:${q}`
  );

  console.log(`the answer is ${ans}`);

  // const notesWithSummaries = await addSummaries(notes)

  // writeOutSummaries(notesWithSummaries)
};

// const question = "what's my t-mobile account number";
// const question = "where were places to get lasik";
const question = "what's my microsoft stock account number";
// main(question);


const indexFile = async () => { 
    
    // get the first and second arg from the command line
    const fileName = process.argv[2];

    // read the file into contents
    const content = await readFile(fileName, "utf8");

    const embedding = await getEmbedding(content)


    const res = await upsert(fileName, embedding);

    console.log(res)





 }

//  indexFile()
