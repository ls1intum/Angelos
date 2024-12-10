from typing import List, Optional

from app.managers.weaviate_manager import WeaviateManager
from app.injestion.document_splitter import DocumentSplitter
from app.data.knowledge_base_requests import AddWebsiteRequest, EditDocumentRequest, EditSampleQuestionRequest, EditWebsiteRequest, AddDocumentRequest, AddSampleQuestionRequest, RefreshContentRequest
from app.data.database_requests import DatabaseDocument, DatabaseDocumentMetadata, DatabaseSampleQuestion

class InjestionHandler:
    def __init__(self, weaviate_manager: WeaviateManager, document_splitter: DocumentSplitter):
        self.weaviate_manager = weaviate_manager
        self.document_splitter = document_splitter
        
    def add_website(self, website: AddWebsiteRequest):
        website_docs: List[DatabaseDocument] = []
        if website.type == "CIT":
            chunks = self.document_splitter.split_cit_content(website.content)
            for chunk in chunks:
                website_docs.append(
                    DatabaseDocument(
                        id=website.id,
                        content=chunk,
                        link=website.link,
                        study_programs=self.prepare_study_programs(website.studyPrograms)
                    )
                )
        else:
            chunks = self.document_splitter.split_tum_content(website.content)
            for chunk in chunks:
                website_docs.append(
                    DatabaseDocument(
                        id=website.id,
                        content=chunk,
                        link=website.link,
                        study_programs=self.prepare_study_programs(website.studyPrograms)
                    )
                )
        self.weaviate_manager.add_documents(website_docs)
        
    def add_document(self, document: AddDocumentRequest):
        website_docs: List[DatabaseDocument] = []
        chunks = self.document_splitter.split_pdf_document(document.content)
        for chunk in chunks:
            website_docs.append(
                DatabaseDocument(
                    id=document.id,
                    content=chunk,
                    study_programs=self.prepare_study_programs(document.studyPrograms)
                )
            )
        self.weaviate_manager.add_documents(website_docs)
        
    def update_database_document(self, id: int, metadata: DatabaseDocumentMetadata):
        self.weaviate_manager.update_documents(id, self.prepare_study_programs(metadata.study_programs))
        
    def refresh_content(self, id: int, content: str):
        metadata: Optional[DatabaseDocumentMetadata] = self.weaviate_manager.delete_by_kb_id(kb_id=id, return_metadata=True)
        if metadata is not None:
            website_docs: List[DatabaseDocument] = []
            if metadata.link is None:
                chunks = self.document_splitter.split_pdf_document(content)
                for chunk in chunks:
                    website_docs.append(
                        DatabaseDocument(
                            id=id,
                            content=chunk,
                            study_programs=self.prepare_study_programs(metadata.study_programs)
                        )
                    )
            else:
                if "cit.tum.de" in metadata.link:
                    chunks = self.document_splitter.split_cit_content(content)
                    for chunk in chunks:
                        website_docs.append(
                            DatabaseDocument(
                                id=id,
                                content=chunk,
                                link=metadata.link,
                                study_programs=self.prepare_study_programs(metadata.study_programs)
                            )
                        )
                else:
                    chunks = self.document_splitter.split_tum_content(content)
                    for chunk in chunks:
                        website_docs.append(
                            DatabaseDocument(
                                id=id,
                                content=chunk,
                                link=metadata.link,
                                study_programs=self.prepare_study_programs(metadata.study_programs)
                            )
                        )
            self.weaviate_manager.add_documents(website_docs)
            
    def delete_document(self, id: str):
        self.weaviate_manager.delete_by_kb_id(kb_id=id, return_metadata=False)
        
    def add_sample_question(self, sample_question: AddSampleQuestionRequest):
        database_sq = DatabaseSampleQuestion(
            id=sample_question.id,
            topic=sample_question.topic,
            question=sample_question.question,
            answer=sample_question.answer,
            study_programs=sample_question.studyPrograms
        )
        self.weaviate_manager.add_sample_question(database_sq)
        
    def update_sample_question(self, kb_id: int, sample_question: EditSampleQuestionRequest):
        database_sq = DatabaseSampleQuestion(
            id=kb_id,
            topic=sample_question.topic,
            question=sample_question.question,
            answer=sample_question.answer,
            study_programs=sample_question.studyPrograms
        )
        self.weaviate_manager.update_sample_question(database_sq)
        
    def delete_sample_question(self, id: int):
        self.weaviate_manager.delete_sample_question(id=id)
    
    # Handle content not specific to study programs
    def prepare_study_programs(self, study_programs: List[str]) -> List[str]:
        if len(study_programs) == 0:
            return ["general"]
        else:
            return study_programs
        
        
            